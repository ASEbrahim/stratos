"""
URL validation utilities for SSRF prevention.

Blocks requests to internal/private IP ranges, non-HTTP schemes,
and performs DNS resolution before connecting to prevent DNS rebinding.
"""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger("STRAT_OS")

# Allowed URL schemes
_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Private/reserved IP networks (RFC 1918, RFC 4193, loopback, link-local, etc.)
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),         # "This" network
    ipaddress.ip_network("10.0.0.0/8"),         # RFC 1918
    ipaddress.ip_network("100.64.0.0/10"),      # Carrier-grade NAT
    ipaddress.ip_network("127.0.0.0/8"),        # Loopback
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local
    ipaddress.ip_network("172.16.0.0/12"),      # RFC 1918
    ipaddress.ip_network("192.0.0.0/24"),       # IETF protocol assignments
    ipaddress.ip_network("192.0.2.0/24"),       # TEST-NET-1
    ipaddress.ip_network("192.88.99.0/24"),     # 6to4 relay
    ipaddress.ip_network("192.168.0.0/16"),     # RFC 1918
    ipaddress.ip_network("198.18.0.0/15"),      # Benchmarking
    ipaddress.ip_network("198.51.100.0/24"),    # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),     # TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),        # Multicast
    ipaddress.ip_network("240.0.0.0/4"),        # Reserved
    ipaddress.ip_network("255.255.255.255/32"), # Broadcast
    # IPv6
    ipaddress.ip_network("::1/128"),            # Loopback
    ipaddress.ip_network("fc00::/7"),           # Unique local (RFC 4193)
    ipaddress.ip_network("fe80::/10"),          # Link-local
    ipaddress.ip_network("::ffff:0:0/96"),      # IPv4-mapped IPv6
]


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is in a private/reserved range."""
    try:
        addr = ipaddress.ip_address(ip_str)
        for network in _BLOCKED_NETWORKS:
            if addr in network:
                return True
        return False
    except ValueError:
        return True  # If we can't parse it, block it


def validate_url(url: str) -> tuple:
    """Validate a URL for safe server-side fetching.

    Returns (is_safe, error_message). If is_safe is True, error_message is None.
    Performs DNS resolution to prevent DNS rebinding attacks.
    """
    if not url:
        return False, "Empty URL"

    parsed = urlparse(url)

    # 1. Scheme check
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False, f"Blocked scheme: {parsed.scheme}"

    # 2. Hostname check
    hostname = parsed.hostname
    if not hostname:
        return False, "No hostname in URL"

    # Block localhost aliases (with and without brackets)
    _localhost_aliases = {"localhost", "0.0.0.0", "[::1]", "::1", "[0:0:0:0:0:0:0:1]"}
    if hostname.lower() in _localhost_aliases:
        return False, "Blocked: localhost"

    # Block userinfo in URL (http://evil@internal/) — urlparse allows it
    if parsed.username is not None:
        return False, "Blocked: URL contains userinfo"

    # 3. Try to parse hostname as raw IP — catches hex (0x7f000001),
    #    octal (0177.0.0.1), and decimal (2130706433) bypass attempts.
    #    Python's ipaddress handles these forms.
    bare_host = hostname.strip("[]")  # strip IPv6 brackets
    try:
        direct_addr = ipaddress.ip_address(bare_host)
        for network in _BLOCKED_NETWORKS:
            if direct_addr in network:
                return False, f"Blocked: IP {bare_host} is in a private range"
    except ValueError:
        pass  # Not a literal IP — will be resolved via DNS below

    # 4. DNS resolution — resolve BEFORE connecting to prevent DNS rebinding
    try:
        addr_infos = socket.getaddrinfo(hostname, parsed.port or 80, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        logger.warning(f"URL validation: DNS resolution failed for {hostname}")
        return False, f"DNS resolution failed for {hostname}"

    if not addr_infos:
        return False, f"DNS returned no results for {hostname}"

    for family, type_, proto, canonname, sockaddr in addr_infos:
        ip = sockaddr[0]
        if _is_private_ip(ip):
            logger.warning(f"URL validation: {hostname} resolves to private IP {ip}")
            return False, f"Blocked: {hostname} resolves to private IP {ip}"

    return True, None
