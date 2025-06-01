# DevClub Cybersecurity Course – 5 Week Plan

Welcome to the cybersecurity vertical of the CSoT part by DevClub, IIT Delhi. This short course is designed to provide a hands-on, engaging introduction to practical cybersecurity through the lens of offensive security techniques. Over the next five weeks, you’ll explore key areas such as Linux fundamentals, cryptography, web and network security, and exploitation techniques—culminating in weekly CTF challenges to apply what you've learned.

While the course emphasizes offensive techniques (red teaming) to make learning fun and interactive, understanding these methods is essential for anyone looking to defend systems effectively. Whether you're interested in ethical hacking or system security and auditing (blue teaming), this foundation will help you recognize, exploit, and ultimately mitigate real-world vulnerabilities.

**Start Date:** 27th May, 2025
**Format:** Weekly resource bundles + self-paced study + weekend CTF challenges  
**Audience:** IIT Delhi students

---

## [Week 1 – Linux Fundamentals & Cybersecurity Onboarding](Week1/)

**Topics:**
- Introduction to cybersecurity and CTFs (types, platforms, mindset)
- Setting up accounts on HackTheBox, TryHackMe, PortSwigger, CTFtime
- Installing Kali Linux (WSL/VirtualBox/Dualboot)
- Linux command line essentials (navigation, permissions, pipes, redirection)
- Basic bash scripting (variables, loops, conditionals)
- Basic reconnaissance commands (`netstat`, `ps`, `ss`)

**Assignments:**  
- Linux basics scavenger hunt (TryHackMe or custom)  
- Write bash scripts for file search and info gathering  
- Setup and explore cybersecurity platforms

**CTF Challenge:**  
- Linux command usage and simple scripting challenges

---

## Week 2 – Cryptography, Steganography & Basic Forensics

**Topics:**  
- Classical ciphers (Caesar, Vigenère) and encoding schemes (Base64, Hex) and their practical uses
- Hash functions & cracking basics (MD5, SHA family) with tools like Hashcat  
- CyberChef for encoding/decoding and crypto puzzles  
- Introduction to steganography (image/audio file hiding, tools like Steghide, zsteg)  
- Basic forensic techniques: metadata extraction, simple file carving, pcap analysis
- How audit systems and antiviruses provide real time safety using hashes and metadata
- Hands-on tools for crypto and forensic challenges (Binwalk, strings, foremost)  

**Assignments:**  
- Solve cipher and encoding challenges using CyberChef and manual tools  
- Crack simple hashes and practice hash identification  
- Stego challenge: extract hidden data from images/audio  
- Forensics challenge: extract metadata and recover hidden info from files and network captures

**CTF Challenge:**  
- Mixed crypto, stego, and forensic challenges

---

## Week 3 – Networking, Reconnaissance & OSINT

**Topics:**  
- Networking fundamentals: TCP/IP, ports, protocols  
- Network scanning and enumeration with `nmap`, `netcat`  
- DNS and subdomain enumeration (`whois`, `dig`, `amass`)  
- OSINT techniques: Google dorking, metadata extraction, username hunting  
- Automating reconnaissance with scripts
- Understanding how recon techniques are used in security audits (e.g., asset discovery, service fingerprinting, attack surface mapping)

**Assignments:**  
- Scan target network and enumerate services  
- Perform OSINT on a mock target and create a report  
- Write scripts to automate recon tasks

**CTF Challenge:** 
- Recon and network scanning challenges

---

## Week 4 – Web Security Fundamentals

**Topics:**  
- HTTP protocol, cookies, sessions, headers  
- OWASP Top 10 overview (SQLi, XSS, CSRF, authentication flaws, IDOR)  
- Burp Suite basics: intercepting and tampering requests  
- Hands-on labs on Juice Shop and PortSwigger Web Academy  
- Introduction to API security testing
- Introduction to web vulnerability scanners and auditing tools: e.g., OWASP ZAP, Nikto, Burp Suite Pro scanning features
- How automated tools are used by auditors to detect vulnerabilities (misconfigurations, exposed endpoints, outdated software)

**Assignments:**  
- Complete beginner-level web security labs  
- Practice manual request tampering using Burp Suite  
- Write scripts automating web attacks

**CTF Challenge:**  
- Web exploitation challenges (SQLi, XSS, session hijacking)

---

## Week 5 – Privilege Escalation, Post-Exploitation & Final CTF

**Topics:**  
- Linux privilege escalation: SUID, cron jobs, PATH hijacking, GTFOBins  
- Basics of Windows privilege escalation (optional)  
- Post-exploitation concepts: persistence, pivoting, data exfiltration  
- Explore how attackers bypass or tamper with logging and how defenders can detect that
- Tools: Metasploit basics, netcat, tcpdump  
- Advanced scripting: Python for pentesting and automation

**Assignments:**  
- Privilege escalation exercises on Linux VM  
- Write Python scripts for post-exploitation tasks  
- Prepare for final CTF challenge

**Final CTF Challenge:** 
- 8–10 multi-topic challenges covering all course topics  
- Team-based timed event with leaderboard and writeup session

---

## Conclusion

By the end of this 5-week journey, you’ll have built a strong foundation in key cybersecurity concepts with a practical, hands-on understanding of how real-world attacks are carried out and mitigated. This course aims to equip you not just with technical skills but with the mindset to approach problems like a security professional — whether as an ethical hacker, red teamer, or systems auditor.

Understanding offensive security is a critical step toward building secure systems. Through this course, you’ll gain insight into how vulnerabilities are discovered and exploited, which is essential for anyone aiming to build, defend, or audit secure infrastructure. Whether you continue toward bug bounty hunting, competitive CTFs, or defensive security roles, you now have the core tools and thinking patterns to go deeper.

The challenges won’t stop here — but neither will your ability to solve them.

