from fastapi import APIRouter, HTTPException
from scapy.all import ARP, Ether, srp
import socket

router = APIRouter()

@router.get("/scan_network")
async def scan_network(ip_range: str):
    devices = scan_for_devices(ip_range)
    ip_cameras = []

    for device in devices:
        ip = device['ip']
        if check_ip_camera(ip):
            ip_cameras.append(ip)

    if not ip_cameras:
        raise HTTPException(status_code=404, detail="No IP cameras found")

    return {"ip_cameras": ip_cameras}

def scan_for_devices(ip_range):
    arp = ARP(pdst=ip_range)
    ether = Ether(dst="ff:ff:ff:ff:ff:ff")
    packet = ether/arp
    result = srp(packet, timeout=3, verbose=0)[0]

    devices = []
    for sent, received in result:
        devices.append({'ip': received.psrc, 'mac': received.hwsrc})

    return devices

def check_ip_camera(ip):
    try:
        ports = [80, 443, 554,]
        for port in ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex((ip, port)) == 0:
                    return True
    except Exception as e:
        print(f"Error checking {ip}: {e}")
    return False
