/*
 * beam_controller.ino
 * Smart Adaptive Headlight — Arduino Firmware
 *
 * Parses binary serial packets from the Jetson/RPi, drives PCA9685
 * PWM channels for matrix LED zones and servos for beam pan/tilt.
 *
 * SAFETY-CRITICAL: Watchdog fail-safe forces Low Beam + center servo
 * if no valid packet is received within WATCHDOG_TIMEOUT_MS.
 *
 * Packet format (from host):
 *   [0xAA] [ZONE_COUNT] [B0..B(N-1)] [PAN] [TILT] [XOR_CHECKSUM]
 *
 * Hardware:
 *   - PCA9685 16-channel PWM driver (I2C addr 0x40)
 *   - MOSFET-switched LED zones on PCA9685 channels 0..N-1
 *   - Pan servo on PCA9685 channel 14
 *   - Tilt servo on PCA9685 channel 15
 */

#include <Wire.h>

// ── Configuration ──────────────────────────────────────────────
#define SYNC_BYTE           0xAA
#define MAX_ZONES           16
#define WATCHDOG_TIMEOUT_MS 1000    // Force low beam if no packet for 1s
#define SERIAL_BAUD         115200

// PCA9685 registers
#define PCA9685_ADDR        0x40
#define PCA9685_MODE1       0x00
#define PCA9685_PRESCALE    0xFE
#define PCA9685_LED0_ON_L   0x06

// Servo channels on PCA9685
#define PAN_CHANNEL         14
#define TILT_CHANNEL        15

// Servo pulse range (in PCA9685 ticks, 4096 scale)
#define SERVO_MIN           150     // ~0.5ms pulse
#define SERVO_MAX           600     // ~2.5ms pulse

// Slew rate limiting
#define MAX_BRIGHTNESS_STEP 15      // Max PWM change per cycle
#define MAX_SERVO_STEP      5       // Max servo ticks per cycle
#define CONTROL_LOOP_MS     20      // Control loop period

// Low beam safety defaults
#define LOW_BEAM_PWM        25      // ~10% brightness
#define PAN_CENTER          90
#define TILT_CENTER         90

// ── State ──────────────────────────────────────────────────────
uint8_t zoneCount = 8;
uint8_t targetBrightness[MAX_ZONES];
uint8_t currentBrightness[MAX_ZONES];
uint8_t targetPan = PAN_CENTER;
uint8_t targetTilt = TILT_CENTER;
uint16_t currentPanTicks;
uint16_t currentTiltTicks;

unsigned long lastValidPacketMs = 0;
bool watchdogTripped = false;

// Receive buffer
#define RX_BUF_SIZE 64
uint8_t rxBuf[RX_BUF_SIZE];
uint8_t rxIdx = 0;
enum RxState { WAIT_SYNC, READ_PAYLOAD };
RxState rxState = WAIT_SYNC;
uint8_t expectedLen = 0;

// ── PCA9685 Driver ─────────────────────────────────────────────

void pca9685_write(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(PCA9685_ADDR);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission();
}

void pca9685_init() {
    Wire.begin();
    pca9685_write(PCA9685_MODE1, 0x10); // Sleep
    // Set PWM frequency ~60Hz for servos (prescale = 100)
    pca9685_write(PCA9685_PRESCALE, 100);
    pca9685_write(PCA9685_MODE1, 0x20); // Auto-increment, no sleep
    delay(5);
}

void pca9685_setPWM(uint8_t channel, uint16_t on, uint16_t off) {
    uint8_t reg = PCA9685_LED0_ON_L + 4 * channel;
    Wire.beginTransmission(PCA9685_ADDR);
    Wire.write(reg);
    Wire.write(on & 0xFF);
    Wire.write(on >> 8);
    Wire.write(off & 0xFF);
    Wire.write(off >> 8);
    Wire.endTransmission();
}

void setZonePWM(uint8_t zone, uint8_t brightness) {
    // Map 0-255 brightness to 0-4095 PCA9685 range
    uint16_t pwm = (uint16_t)brightness * 16;
    if (pwm > 4095) pwm = 4095;
    pca9685_setPWM(zone, 0, pwm);
}

uint16_t angleToPWMTicks(uint8_t angle) {
    // Map 0-180 degrees to SERVO_MIN..SERVO_MAX
    return map(angle, 0, 180, SERVO_MIN, SERVO_MAX);
}

void setServo(uint8_t channel, uint16_t ticks) {
    pca9685_setPWM(channel, 0, ticks);
}

// ── Slew Rate Limiting ─────────────────────────────────────────

int16_t slewLimit(int16_t current, int16_t target, int16_t maxStep) {
    int16_t diff = target - current;
    if (diff > maxStep) return current + maxStep;
    if (diff < -maxStep) return current - maxStep;
    return target;
}

// ── Packet Parsing ─────────────────────────────────────────────

bool parsePacket(uint8_t *buf, uint8_t len) {
    // buf[0] = ZONE_COUNT
    // buf[1..N] = brightness values
    // buf[N+1] = PAN
    // buf[N+2] = TILT
    // buf[N+3] = XOR checksum (over bytes 0..N+2)

    uint8_t nZones = buf[0];
    if (nZones == 0 || nZones > MAX_ZONES) return false;

    uint8_t expectedPayloadLen = nZones + 3 + 1; // count + zones + pan + tilt + checksum
    if (len != expectedPayloadLen) return false;

    // Verify checksum
    uint8_t xorCheck = 0;
    for (uint8_t i = 0; i < len - 1; i++) {
        xorCheck ^= buf[i];
    }
    if (xorCheck != buf[len - 1]) return false;

    // Parse
    zoneCount = nZones;
    for (uint8_t i = 0; i < nZones; i++) {
        targetBrightness[i] = buf[1 + i];
    }
    targetPan = buf[1 + nZones];
    targetTilt = buf[2 + nZones];

    return true;
}

void processSerial() {
    while (Serial.available()) {
        uint8_t b = Serial.read();

        switch (rxState) {
            case WAIT_SYNC:
                if (b == SYNC_BYTE) {
                    rxIdx = 0;
                    rxState = READ_PAYLOAD;
                }
                break;

            case READ_PAYLOAD:
                rxBuf[rxIdx++] = b;

                // After first byte (zone count), we know expected length
                if (rxIdx == 1) {
                    expectedLen = b + 3 + 1; // zones + pan + tilt + checksum + count byte
                    if (expectedLen > RX_BUF_SIZE) {
                        rxState = WAIT_SYNC;  // Invalid, reset
                    }
                }

                if (rxIdx >= expectedLen) {
                    if (parsePacket(rxBuf, rxIdx)) {
                        lastValidPacketMs = millis();
                        watchdogTripped = false;
                    }
                    rxState = WAIT_SYNC;
                }
                break;
        }
    }
}

// ── Watchdog Fail-Safe ─────────────────────────────────────────

void watchdogCheck() {
    if (millis() - lastValidPacketMs > WATCHDOG_TIMEOUT_MS) {
        if (!watchdogTripped) {
            watchdogTripped = true;
            Serial.println("WDT:TRIPPED");
        }
        // Force low beam + center servos
        for (uint8_t i = 0; i < MAX_ZONES; i++) {
            targetBrightness[i] = LOW_BEAM_PWM;
        }
        targetPan = PAN_CENTER;
        targetTilt = TILT_CENTER;
    }
}

// ── Control Loop ───────────────────────────────────────────────

void updateOutputs() {
    // Slew-rate limited brightness update
    for (uint8_t i = 0; i < zoneCount; i++) {
        currentBrightness[i] = (uint8_t)slewLimit(
            (int16_t)currentBrightness[i],
            (int16_t)targetBrightness[i],
            MAX_BRIGHTNESS_STEP
        );
        setZonePWM(i, currentBrightness[i]);
    }

    // Slew-rate limited servo update
    uint16_t targetPanTicks = angleToPWMTicks(targetPan);
    uint16_t targetTiltTicks = angleToPWMTicks(targetTilt);

    currentPanTicks = (uint16_t)slewLimit(
        (int16_t)currentPanTicks,
        (int16_t)targetPanTicks,
        MAX_SERVO_STEP
    );
    currentTiltTicks = (uint16_t)slewLimit(
        (int16_t)currentTiltTicks,
        (int16_t)targetTiltTicks,
        MAX_SERVO_STEP
    );

    setServo(PAN_CHANNEL, currentPanTicks);
    setServo(TILT_CHANNEL, currentTiltTicks);
}

// ── Arduino Entry Points ───────────────────────────────────────

void setup() {
    Serial.begin(SERIAL_BAUD);

    // Initialize PCA9685
    pca9685_init();

    // Set initial state: low beam, centered
    for (uint8_t i = 0; i < MAX_ZONES; i++) {
        targetBrightness[i] = LOW_BEAM_PWM;
        currentBrightness[i] = LOW_BEAM_PWM;
    }
    currentPanTicks = angleToPWMTicks(PAN_CENTER);
    currentTiltTicks = angleToPWMTicks(TILT_CENTER);

    lastValidPacketMs = millis();

    updateOutputs();

    Serial.println("BEAM_CTRL:READY");
}

void loop() {
    static unsigned long lastLoopMs = 0;
    unsigned long now = millis();

    // Process incoming serial data
    processSerial();

    // Fixed-rate control loop
    if (now - lastLoopMs >= CONTROL_LOOP_MS) {
        lastLoopMs = now;

        // Check watchdog
        watchdogCheck();

        // Update PWM outputs with slew limiting
        updateOutputs();
    }
}
