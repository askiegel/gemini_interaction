# Mini Pupper 2 Wi-Fi Configuration

> Status: Verified on hardware

---

## Purpose

The Mini Pupper 2 is configured to use an iPhone Personal Hotspot for mobile operation and the home Wi-Fi network as a fallback.

## Configured Networks

| Priority | NetworkManager connection | Purpose |
|---:|---|---|
| 100 | `iPhone-Tony-manual` | Preferred mobile connection |
| 50 | `netplan-wlan0-NETGEAR94` | Home and lab fallback |

The higher-priority iPhone connection is preferred when it is available.

## Verified iPhone Hotspot Settings

- SSID: `iPhone-Tony`
- Security: WPA2 Personal
- Maximize Compatibility: Enabled
- NetworkManager profile: `iPhone-Tony-manual`

Do not store the hotspot password in this repository.

## Verified Hotspot Connection

The following connection was verified on hardware:

- Active connection: `iPhone-Tony-manual`
- Wireless interface: `wlan0`
- SSID: `iPhone-Tony`
- Frequency: 2437 MHz
- Assigned address: `172.20.10.2/28`
- Default gateway: `172.20.10.1`
- Signal during validation: approximately `-60 dBm`

The hotspot BSSID may change and must not be permanently locked in the NetworkManager profile.

## Verify the Current Connection

Run these commands on the Mini Pupper:

    echo "===== ACTIVE CONNECTION ====="
    nmcli -f NAME,DEVICE connection show --active

    echo
    echo "===== WIFI LINK ====="
    iw dev wlan0 link

    echo
    echo "===== IP ADDRESS ====="
    ip -4 addr show wlan0

    echo
    echo "===== DEFAULT ROUTE ====="
    ip route | grep '^default'

Expected values while connected to the iPhone hotspot:

- Connection: `iPhone-Tony-manual`
- SSID: `iPhone-Tony`
- Address range: `172.20.10.x`
- Default gateway: `172.20.10.1`

Expected values while connected to the home network:

- Connection: `netplan-wlan0-NETGEAR94`
- Known validation address: `192.168.68.124`

The DHCP address can change, so verify it whenever the robot changes networks.

## Important Reboot Requirement

After creating or substantially changing a Wi-Fi profile, reboot the Mini Pupper before concluding that the profile does not work.

During hardware validation:

- The hotspot was visible.
- The password was correctly stored.
- NetworkManager reported that the required secrets existed.
- Association attempts still failed.
- Rebooting the Mini Pupper cleared the stale wireless state.
- The robot connected successfully to the iPhone hotspot after reboot.

The failure was consistent with stale NetworkManager, `wpa_supplicant`, or Broadcom wireless-driver state.

Reboot with:

    sudo reboot

Keep the iPhone Personal Hotspot settings screen open while the Mini Pupper starts.

## Troubleshooting

Show all configured connections:

    nmcli connection show

Show active connections:

    nmcli -f NAME,DEVICE connection show --active

Rescan and list nearby Wi-Fi networks:

    sudo nmcli device wifi rescan
    nmcli device wifi list

Inspect the current wireless link:

    iw dev wlan0 link

Inspect the IP address:

    ip -4 addr show wlan0

Inspect the default route:

    ip route | grep '^default'

Inspect recent NetworkManager messages:

    journalctl -u NetworkManager -n 100 --no-pager

Inspect the wireless driver:

    ethtool -i wlan0

Expected driver:

    brcmfmac

## Avoid BSSID Locking

Do not permanently configure a BSSID for the iPhone hotspot.

The hotspot BSSID changed during testing:

- Earlier observed BSSID: `1E:BA:67:69:84:02`
- Later verified BSSID: `86:8A:6A:00:6D:55`

A stale BSSID lock caused NetworkManager to report `ssid-not-found`, even though the hotspot SSID was visible.

The profile should select the hotspot by SSID rather than by a fixed BSSID.

## Verified Platform

- Robot: Mini Pupper 2
- Operating system: Ubuntu 22.04 LTS
- Kernel: `5.15.0-1103-raspi`
- Wi-Fi driver: `brcmfmac`
- NetworkManager: `1.36.6`
- wpa_supplicant: `2.10`

## Safety and Access Note

Changing networks can terminate the active SSH session.

Before testing network switching:

1. Stop the robot safely.
2. Record the current IP address.
3. Confirm that another access method is available.
4. Expect the SSH session to disconnect when the active network changes.
