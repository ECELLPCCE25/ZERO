from fastapi import APIRouter, HTTPException
from scapy.all import ARP, Ether, srp
import socket
import netifaces

router = APIRouter()

@router.get("/scan_network")
async def scan_network():
    local_ip = get_local_ip()
    if not local_ip:
        raise HTTPException(status_code=500, detail="Unable to determine local IP address")

    # Define the IP ranges to scan
    ip_ranges = [
        f"{local_ip}/24",
        # Add more IP ranges if needed
    ]

    ip_cameras = []

    for ip_range in ip_ranges:
        devices = scan_for_devices(ip_range)
        for device in devices:
            ip = device['ip']
            if check_ip_camera(ip):
                ip_cameras.append(ip)

    if not ip_cameras:
        raise HTTPException(status_code=404, detail="No IP cameras found")

    return {"ip_cameras": ip_cameras}

def get_local_ip():
    interfaces = netifaces.interfaces()
    for interface in interfaces:
        addrs = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addrs:
            for addr in addrs[netifaces.AF_INET]:
                if addr['addr'] != '127.0.0.1':
                    return addr['addr']
    return None

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
        # Include additional ports commonly used by IP camera apps on Android
        ports = [80, 443, 554, 8000, 8080, 8554, 8888, 5000, 5001, 5002, 5003, 5004, 5005, 4747]
        for port in ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex((ip, port)) == 0:
                    return True
    except Exception as e:
        print(f"Error checking {ip}: {e}")
    return False