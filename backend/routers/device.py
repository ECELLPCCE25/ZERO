from fastapi import APIRouter, HTTPException, FastAPI
from scapy.all import ARP, Ether, srp
import socket
import netifaces
import time
import threading

# Create a FastAPI app instance (needed for cache initialization - though we'll remove cache specific init)
# app = FastAPI() # Removed as it's not needed for this router file anymore

router = APIRouter()

# Manual Cache implementation
# Dictionary to store cached results: {cache_key: {"result": result, "timestamp": timestamp}}
network_scan_cache = {}
# Cache expiration time in seconds
CACHE_EXPIRY_SECONDS = 20


def get_cache_key():
    """Generates a cache key based on the local IP."""
    local_ip = get_local_ip()
    # Using the local IP in the key ensures different networks have different caches
    return f"scan_network:{local_ip}"


def is_cache_valid(cache_key):
    """Checks if the cache for the given key is still valid."""
    if cache_key in network_scan_cache:
        cached_data = network_scan_cache[cache_key]
        # Check if the cache has expired
        if time.time() - cached_data["timestamp"] < CACHE_EXPIRY_SECONDS:
            print(f"Cache hit for {cache_key}. Returning cached data.")
            return True
        else:
            print(f"Cache expired for {cache_key}. Performing new scan.")
            # Remove expired cache entry
            del network_scan_cache[cache_key]
    return False


def update_cache(cache_key, result):
    """Updates the cache with a new result and timestamp."""
    network_scan_cache[cache_key] = {"result": result, "timestamp": time.time()}
    print(f"Cache updated for {cache_key}.")


@router.get("/scan_network")
async def scan_network():
    """
    Scans the local network for potential IP cameras.
    Results are cached manually with a refresh interval.
    """
    cache_key = get_cache_key()

    # Check if cache is valid
    if is_cache_valid(cache_key):
        return network_scan_cache[cache_key]["result"]

    # If cache is not valid, perform a new scan
    print("Performing network scan...")
    local_ip = get_local_ip()
    if not local_ip:
        raise HTTPException(
            status_code=500, detail="Unable to determine local IP address"
        )

    ip_ranges = [
        f"{local_ip}/24",
    ]

    ip_cameras = []

    for ip_range in ip_ranges:
        devices = scan_for_devices(ip_range)
        for device in devices:
            ip = device["ip"]
            # Note: check_ip_camera can be slow, consider making it async or
            # running it in a thread pool if scanning many devices.
            if check_ip_camera(ip):
                ip_cameras.append(ip)

    result = {"ip_cameras": ip_cameras}

    # Update the cache with the new result
    update_cache(cache_key, result)

    if not ip_cameras:
        return {"ip_cameras": []}

    return result


def get_local_ip():
    """
    Attempts to find the local machine's non-loopback IPv4 address.
    """
    interfaces = netifaces.interfaces()
    for interface in interfaces:
        try:
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    if addr["addr"] != "127.0.0.1":
                        return addr["addr"]
        except ValueError:
            continue
    return None


def scan_for_devices(ip_range):
    """
    Performs an ARP scan on the specified IP range to discover active devices.
    Requires root/administrator privileges.
    """
    print(f"Scanning IP range: {ip_range}")
    try:
        # Using a shorter timeout for faster scanning, adjust if needed
        ans, unans = srp(
            Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=ip_range), timeout=1, verbose=0
        )

        devices = []
        for sent, received in ans:
            devices.append({"ip": received.psrc, "mac": received.hwsrc})

        return devices
    except PermissionError:
        print("Permission denied. ARP scan requires root/administrator privileges.")
        return []
    except Exception as e:
        print(f"Error during ARP scan: {e}")
        return []


def check_ip_camera(ip):
    """
    Attempts to determine if a given IP address belongs to a potential IP camera
    by checking for open ports.
    """
    # print(f"Checking IP: {ip}") # Uncomment for debugging
    try:
        ports = [
            80,
            443,
            554,
            8000,
            8080,
            8554,
            8888,
            5000,
            5001,
            5002,
            5003,
            5004,
            5005,
            4747,
        ]
        for port in ports:
            try:
                with socket.create_connection(
                    (ip, port), timeout=0.3
                ) as s:  # Reduced timeout
                    # If connection is successful, it's likely a device, could be a camera
                    # print(f"Port {port} open on {ip}") # Uncomment for debugging
                    return True
            except (socket.timeout, ConnectionRefusedError, OSError):
                # print(f"Port {port} closed or connection refused on {ip}") # Uncomment for debugging
                continue  # Try next port
        # print(f"No common camera ports open on {ip}") # Uncomment for debugging
        return False
    except Exception:
        # print(f"Error checking IP {ip}: {e}") # Uncomment for debugging
        return False
