# Hardware Setup Guide

## System Hardware Specification

| Component | Model / Specification | Interface |
|-----------|------------------------|-----------|
| Compute Host | NVIDIA Jetson Orin Nano / AGX Orin | Ubuntu 22.04 LTS (JetPack 5.1/6.0) |
| Camera | Sony IMX327 Low-Light Automotive Sensor | CSI-2 / USB 3.0 (1080p @ 30 FPS) |
| Microcontroller | Arduino Uno / Nano / Mega | USB Serial (/dev/ttyACM0 @ 115200) |
| Matrix LED Array | 8-Zone High-Power LED Driver Board | MOSFET / PWM channels 2-9 |

## Pinout Map

```
Arduino MCU:
Pin 2  --> Matrix Zone 0 (Far Left)
Pin 3  --> Matrix Zone 1
Pin 4  --> Matrix Zone 2
Pin 5  --> Matrix Zone 3 (Center Left)
Pin 6  --> Matrix Zone 4 (Center Right)
Pin 7  --> Matrix Zone 5
Pin 8  --> Matrix Zone 6
Pin 9  --> Matrix Zone 7 (Far Right)
Pin 13 --> Status Indicator LED
```

## Serial Protocol Specification

Frame format transmitted from Jetson to Arduino over Serial:

- **Header**: `0xAA 0x55` (2 bytes)
- **Payload**: `8 x uint8_t` (PWM values 0-255 for zones 0 to 7)
- **Checksum**: XOR of payload bytes (1 byte)
- **Total Frame Length**: 11 bytes
