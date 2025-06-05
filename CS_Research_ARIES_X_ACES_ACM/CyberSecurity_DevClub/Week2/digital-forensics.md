# Digital Forensics Fundamentals

Digital forensics involves collecting, preserving, and analyzing digital evidence to investigate security incidents and data breaches.

---

## File Carving and Recovery

### Understanding Deleted Files

When you delete a file, the OS only removes the directory entry - the actual data remains until overwritten. File carving searches disk images for file signatures to recover deleted files.

### File Signatures (Magic Bytes)

Every file type starts with unique "magic bytes":

```bash
PNG:  89 50 4E 47 (‰PNG)
JPEG: FF D8 FF E0 
PDF:  25 50 44 46 (%PDF)
ZIP:  50 4B 03 04 (PK..)
GIF:  47 49 46 38 (GIF8)
MP3:  49 44 33 (ID3)
```

**Practical File Analysis:**
```bash
# Using online hex editor to identify files
1. Go to https://hexed.it/
2. Upload suspicious file
3. Check first 16 bytes for signature
4. Example: File named "document.txt" but starts with 50 4B = ZIP file!

# File recovery simulation
Scenario: Find JPEG in disk image
Search for: FF D8 FF signature
Found at: Offset 1,250,000 bytes
Extract: 2.3 MB of data after signature
Result: Recovered family_vacation.jpg with GPS metadata
```

**File Recovery Tools:**
- **Foremost**: Searches for headers/footers, organizes by file type
- **PhotoRec**: Specializes in media recovery, works on damaged systems
- **Scalpel**: Advanced carving with custom rules

**Online Tools:**
- **File Signatures Database**: https://www.filesignatures.net/
- **Online Hex Editor**: https://hexed.it/

---

## Network Forensics (PCAP Analysis)

### Understanding Network Captures

PCAP files contain complete network conversations with headers, payloads, and timestamps.

**Example Analysis Process:**
```bash
# Step 1: Protocol Overview
HTTP: 45% (normal web browsing)
HTTPS: 30% (secure traffic)  
SSH: 8% (remote access)
Unknown: 2% (investigate!)

# Step 2: Communication Analysis
Internal: 192.168.1.100
Connected to:
- google.com (legitimate)
- suspicious-domain.com (unknown - red flag!)
- 203.0.113.1 (known malware C&C server)

# Step 3: Content Examination
14:30:15 - HTTP POST to file-sharing.com
Size: 2.3MB upload from 192.168.1.50
User-Agent: Suspicious (doesn't match corporate standard)
Conclusion: Potential data exfiltration
```

**Online PCAP Analysis Tools:**

**PacketTotal** - https://packettotal.com/
- Upload PCAP files for automated analysis
- Threat intelligence and IOC detection
- Detailed security reports

**A-Packets** - https://apackets.com/
- Online Wireshark alternative
- No installation required
- Filter by protocols: http, https, ftp, dns

**CloudShark** - https://www.cloudshark.org/
- Professional web-based analysis
- Collaborative investigation features

**Practical PCAP Analysis:**
```bash
# Upload to PacketTotal for quick analysis
1. Go to https://packettotal.com/
2. Upload your .pcap file
3. Review automated findings:
   - Suspicious IP addresses
   - Unusual protocols
   - Large file transfers
   - Malware signatures

# Manual analysis with A-Packets
1. Filter by protocol types
2. Look for data exfiltration patterns
3. Extract transferred files
4. Analyze payload content
```

---

## Disk and Memory Forensics

### Disk Image Analysis

**Disk images** are exact bit-for-bit copies that preserve all data including deleted files and free space.

**Timeline Analysis Example:**
```bash
2023-05-20 09:15:23 - User login
2023-05-20 09:16:45 - USB device inserted
2023-05-20 09:17:12 - File copied to USB (confidential.docx)
2023-05-20 09:18:33 - Browser opened, file-sharing site visited
2023-05-20 09:19:55 - File uploaded to external site
2023-05-20 09:20:14 - Browser history cleared
2023-05-20 09:20:45 - USB removed, user logout
```

**Windows Registry Evidence:**
```bash
USB Device History:
- Device: SanDisk Cruzer 32GB
- First Connected: 2023-05-15 14:30:22
- Last Connected: 2023-05-20 09:16:45

Recent Documents:
- confidential_budget.xlsx (opened 08:45:12)
- employee_data.csv (opened 09:10:33)
- resignation_letter.docx (created 09:14:55)
```

### Memory Forensics

**Memory contains:**
- Running processes and active network connections
- Passwords and encryption keys in plain text
- Malware that only exists in RAM
- Recently accessed files and applications

**Process Analysis Example:**
```bash
Suspicious Processes:
- svchost.exe (PID: 1337) - Running from unusual location
- backdoor.exe (PID: 3456) - Unknown executable
- cmd.exe (PID: 4567) - Check command history

Active Network Connections:
- chrome.exe → facebook.com (legitimate)
- backdoor.exe → 203.0.113.1:443 (malicious C&C)
```

**Memory Analysis Tools:**
- **Volatility Framework**: https://volatilityfoundation.org/ - Industry standard
- **Memory Analysis Resources**: https://www.memoryanalysis.net/ - Learning materials

---

## Evidence Handling

### Chain of Custody

Legal requirement to document who handled evidence and when.

**Documentation Example:**
```bash
Evidence ID: CASE2023-001-HDD
Description: 500GB laptop hard drive
Collected: Detective Smith, 2023-05-20 10:30:00
Location: Suspect's office
Hash: SHA-256: a1b2c3d4e5f6789...

Access Log:
2023-05-21 09:00 - Analyst Jones - Imaging
2023-05-22 14:30 - Analyst Brown - Analysis
2023-05-23 11:15 - Detective Smith - Review
```

### Evidence Integrity

**Hash Verification:**
```bash
Original evidence: SHA-256: a1b2c3d4e5f6789...
After analysis:    SHA-256: a1b2c3d4e5f6789...
Result: MATCH - Evidence integrity verified ✓
```

**Key Principles:**
1. **Preserve original** - Never modify source evidence
2. **Document everything** - Detailed logs and procedures
3. **Verify integrity** - Use cryptographic hashes
4. **Follow legal procedures** - Admissibility requirements
5. **Think systematically** - What story does evidence tell?

---

## Essential Tools and Resources

### Online Forensics Tools
- **PacketTotal**: https://packettotal.com/ - PCAP analysis
- **A-Packets**: https://apackets.com/ - Online Wireshark
- **HexEd.it**: https://hexed.it/ - Hex editor for file analysis
- **File Signatures**: https://www.filesignatures.net/ - Magic bytes reference

### Learning Resources
- **Autopsy**: https://www.autopsy.com/ - Free forensics platform
- **Digital Corpora**: https://digitalcorpora.org/ - Practice datasets
- **SANS Digital Forensics**: https://www.sans.org/ - Professional training
- **Forensics Wiki**: https://www.forensicswiki.org/ - Knowledge base

### Investigation Workflow
1. **Secure scene** → Prevent evidence destruction
2. **Image devices** → Create forensic copies
3. **Analyze systematically** → Follow procedures  
4. **Document findings** → Detailed reports
5. **Present conclusions** → Clear explanations for non-technical audience

Remember: In real investigations, proper legal procedures and chain of custody are critical for evidence admissibility in court!