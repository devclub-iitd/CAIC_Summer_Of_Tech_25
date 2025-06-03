# DevClub Cybersecurity Course – Week 1 Resources
## Linux Fundamentals & Cybersecurity Onboarding

Welcome to your cybersecurity journey! This week, you'll set up your environment, create accounts on essential platforms, and master Linux fundamentals that form the backbone of cybersecurity work.

---

## Learning Objectives
By the end of this week, you will:
- Understand cybersecurity fundamentals and CTF basics
- Have working accounts on major cybersecurity platforms
- Have a functional Kali Linux environment
- Master essential Linux command line operations
- Write basic bash scripts for automation
- Perform basic system reconnaissance

---

## Essential Platform Setup

### 1. Create Accounts (Do this FIRST!)

**HackTheBox** - Premium penetration testing labs
- Website: https://www.hackthebox.com/
- Create free account → Verify email
- Complete "Starting Point" machines for beginners
- **Tip:** Student accounts get discounts on VIP subscriptions

**TryHackMe** - Beginner-friendly cybersecurity training
- Website: https://tryhackme.com/
- Create free account → Complete profile
- Start with "Complete Beginner" learning path
- **Recommended:** Consider premium for unlimited machine access

**PortSwigger Web Security Academy** - Web application security
- Website: https://portswigger.net/web-security
- Create free account (completely free platform!)
- Download Burp Suite Community Edition
- Complete "Getting Started" section

**CTFtime** - CTF competition tracker
- Website: https://ctftime.org/
- Create account → Join as individual or create team
- Browse upcoming CTF events
- **Note:** This tracks your CTF participation and rankings

---

## Kali Linux Installation

### Option A: WSL (Windows Subsystem for Linux) - Recommended for Windows users

**Prerequisites:** Windows 10/11 with WSL2 enabled

1. **Enable WSL2:**
   ```powershell
   # Run in PowerShell as Administrator
   wsl --install
   # Restart computer if prompted
   ```

2. **Install Kali Linux:**
   ```powershell
   wsl --install -d kali-linux
   ```

3. **Alternative: Microsoft Store**
   - Open Microsoft Store → Search "Kali Linux" → Install
   - Launch and set up username/password

4. **Update and install tools:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y kali-linux-headless
   ```

**Useful WSL Commands:**
- Access Windows files: `/mnt/c/Users/YourUsername/`
- Start Kali: Type `kali` in any terminal
- GUI apps: Install with `sudo apt install kali-win-kex`

### Option B: VirtualBox - Cross-platform solution

1. **Download VirtualBox:**
   - Website: https://www.virtualbox.org/wiki/Downloads
   - Install VirtualBox and Extension Pack

2. **Download Kali Linux VM:**
   - Website: https://www.kali.org/get-kali/#kali-virtual-machines
   - Download VirtualBox 64-bit image
   - **Default credentials:** kali/kali

3. **Import VM:**
   - Open VirtualBox → File → Import Appliance
   - Select downloaded .ova file
   - Allocate at least 4GB RAM, 2 CPU cores

4. **First boot:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

### Option C: Dual Boot - Advanced users only

**Warning:** This can affect your existing OS. Backup everything first!

1. Create Windows recovery drive
2. Download Kali Linux ISO: https://www.kali.org/get-kali/#kali-installer-images
3. Create bootable USB with Rufus/Etcher
4. Partition hard drive (at least 50GB for Kali)
5. Boot from USB and follow installation guide

---

## Linux Command Line Essentials

### Navigation Commands
```bash
# Current directory
pwd

# List files and directories
ls          # Basic listing
ls -la      # Detailed listing with hidden files
ls -lh      # Human-readable file sizes

# Change directory
cd /path/to/directory
cd ~        # Home directory
cd ..       # Parent directory
cd -        # Previous directory

# File operations
mkdir dirname           # Create directory
mkdir -p path/to/dir   # Create nested directories
rmdir dirname          # Remove empty directory
rm filename            # Remove file
rm -rf dirname         # Remove directory and contents (DANGEROUS!)
cp source dest         # Copy file
mv source dest         # Move/rename file
```

### File Viewing and Editing
```bash
# View file contents
cat filename           # Display entire file
less filename          # View file page by page
head filename          # First 10 lines
tail filename          # Last 10 lines
tail -f filename       # Follow file changes (logs)

# Search in files
grep "pattern" filename
grep -r "pattern" directory/    # Recursive search
grep -i "pattern" filename      # Case-insensitive

# Text editors
nano filename          # Beginner-friendly
vim filename           # Advanced (press 'i' to insert, ':wq' to save and quit)
```

### File Permissions
```bash
# View permissions
ls -l filename

# Change permissions
chmod 755 filename     # rwxr-xr-x
chmod +x filename      # Make executable
chmod -w filename      # Remove write permission

# Change ownership
sudo chown user:group filename
```

### Process Management
```bash
# View running processes
ps aux                 # All processes
ps -ef                 # Different format
top                    # Real-time process monitor
htop                   # Enhanced top (if installed)

# Kill processes
kill PID               # Terminate process by ID
killall processname    # Kill all instances
kill -9 PID           # Force kill
```

### Network Commands
```bash
# Network information
ip addr                # IP addresses
ifconfig              # Network interfaces (older)
netstat -tulpn        # Open ports and connections
ss -tulpn             # Modern alternative to netstat

# Network testing
ping google.com       # Test connectivity
nslookup domain.com   # DNS lookup
dig domain.com        # Detailed DNS info
```

### Pipes and Redirection
```bash
# Pipes (send output to another command)
ls -la | grep ".txt"
cat file.txt | grep "error" | wc -l

# Redirection
echo "Hello" > file.txt          # Write to file (overwrite)
echo "World" >> file.txt         # Append to file
command 2> error.log             # Redirect errors
command > output.txt 2>&1        # Redirect both output and errors
```

---

## **Basic Challenge:**
There is a .tar.zx file present in this folder, download it in your linux system and unzip it.
You can use the following command to do the same:
```bash
wget -qO- https://github.com/devclub-iitd/CAIC_Summer_Of_Tech_25/raw/refs/heads/main/CyberSecurity_DevClub/Week1/scavenger_hunt.tar.xz | tar -Jxf -
```
Then using only your terminal, try to find the real hidden flag.

---

## Basic Bash Scripting

### Your First Script
Create a file called `hello.sh`:
```bash
#!/bin/bash
# This is a comment

echo "Hello, DevClub!"
echo "Current user: $(whoami)"
echo "Current directory: $(pwd)"
echo "Current date: $(date)"
```

Make it executable and run:
```bash
chmod +x hello.sh
./hello.sh
```

### Variables
```bash
#!/bin/bash

# Variable assignment (no spaces around =)
name="Alice"
age=25
current_date=$(date)

# Using variables
echo "Name: $name"
echo "Age: ${age} years old"
echo "Today is: $current_date"

# User input
read -p "Enter your name: " user_name
echo "Hello, $user_name!"
```

### Conditionals
```bash
#!/bin/bash

read -p "Enter a number: " num

if [ $num -gt 10 ]; then
    echo "Number is greater than 10"
elif [ $num -eq 10 ]; then
    echo "Number is exactly 10"
else
    echo "Number is less than 10"
fi

# File checks
if [ -f "filename.txt" ]; then
    echo "File exists"
fi

if [ -d "directory" ]; then
    echo "Directory exists"
fi
```

### Loops
```bash
#!/bin/bash

# For loop with range
for i in {1..5}; do
    echo "Number: $i"
done

# For loop with files
for file in *.txt; do
    echo "Processing: $file"
done

# While loop
counter=1
while [ $counter -le 5 ]; do
    echo "Counter: $counter"
    ((counter++))
done
```

### Functions
```bash
#!/bin/bash

# Function definition
greet_user() {
    local name=$1
    echo "Hello, $name! Welcome to DevClub."
}

# Function with return value
calculate_sum() {
    local num1=$1
    local num2=$2
    local sum=$((num1 + num2))
    echo $sum
}

# Call functions
greet_user "Alice"
result=$(calculate_sum 5 3)
echo "Sum: $result"
```

---

## Basic Reconnaissance Commands

### System Information
```bash
# System details
uname -a                  # Kernel information
cat /etc/os-release      # OS information
whoami                   # Current user
id                       # User ID and groups
w                        # Who is logged in
last                     # Login history
```

### Network Reconnaissance
```bash
# Local network info
ip route                 # Routing table
arp -a                   # ARP table
netstat -rn              # Network routes

# Port scanning (basic)
nc -zv target_ip 80      # Check if port 80 is open
nc -zv target_ip 20-25   # Scan port range

# DNS enumeration
nslookup target.com
dig target.com
dig @8.8.8.8 target.com  # Use specific DNS server
```

### File System Reconnaissance
```bash
# Find files
find / -name "*.conf" 2>/dev/null     # Find config files
find /home -name "*.txt" 2>/dev/null  # Find text files
find . -perm -4000 2>/dev/null        # Find SUID files

# Search file contents
grep -r "password" /etc/ 2>/dev/null  # Search for passwords
grep -r "admin" /var/log/ 2>/dev/null # Search logs
```

---

## Week 1 Assignments

### Assignment 1: Linux Basics Scavenger Hunt

**TryHackMe Path:** Complete "Linux Fundamentals" rooms
- Linux Fundamentals Part 1: https://tryhackme.com/room/linuxfundamentalspart1
- Linux Fundamentals Part 2: https://tryhackme.com/room/linuxfundamentalspart2
- Linux Fundamentals Part 3: https://tryhackme.com/room/linuxfundamentalspart3

### Assignment 2: Bash Scripting Challenges

**Challenge 1: File Search Script**
Write a script that:
- Takes a directory path as argument
- Finds all .log files modified in the last 7 days
- Counts total lines in these files
- Outputs a summary report

**Challenge 2: System Info Gatherer**
Create a script that collects:
- System uptime
- Disk usage of home directory
- Number of running processes
- Current network connections
- Save all info to a timestamped report file

**Challenge 3: Password Generator**
Write a script that:
- Generates random passwords of specified length
- Includes uppercase, lowercase, numbers, symbols
- Creates multiple passwords at once
- Saves them to a file with timestamps

### Assignment 3: Platform Exploration

1. **HackTheBox:** Complete "Starting Point" - first 3 machines
2. **TryHackMe:** Complete "Complete Beginner" path introduction
3. **PortSwigger:** Complete "Getting started" labs
4. **CTFtime:** Find and bookmark 3 upcoming beginner-friendly CTFs

---

## Additional Learning Resources

### Free Online Resources
- **Linux Journey:** https://linuxjourney.com/ - Interactive Linux learning
- **OverTheWire Bandit:** https://overthewire.org/wargames/bandit/ - Linux command line game
- **Bash Guide:** https://mywiki.wooledge.org/BashGuide - Comprehensive bash reference
- **ExplainShell:** https://explainshell.com/ - Understand shell commands

### Documentation
- **Linux Man Pages:** Use `man command` for any command documentation
- **Bash Manual:** https://www.gnu.org/software/bash/manual/bash.html
- **Kali Linux Documentation:** https://www.kali.org/docs/

### YouTube Channels
- **NetworkChuck** - Beginner-friendly cybersecurity
- **John Hammond** - CTF walkthroughs and tutorials
- **LiveOverflow** - Advanced security concepts
- **IppSec** - HackTheBox machine walkthroughs

---

## Week 1 CTF Challenge (Day 6)

**Challenge Topics:**
- Linux command line navigation
- File permissions and ownership
- Basic bash scripting
- Process investigation
- Simple reconnaissance

**Format:** 
- 5 challenges, increasing difficulty
- Time limit: 3 hours
- Individual competition with hints available
- Writeups required for solutions

**Preparation Tips:**
- Practice all commands from this guide
- Complete all assignments
- Test your scripts thoroughly

---

## Getting Help

### Community Resources
- **r/cybersecurity** - Reddit community
- **InfoSec Twitter** - Follow @malwareunicorn, @stokfredrik, @gynvael
- **Discord Servers:** Ask for invites to cybersecurity learning servers

### Emergency Commands
```bash
# If you break something:
sudo apt update && sudo apt install --reinstall package_name

# Reset terminal
reset

# Get out of vim (just in case!)
:q!

# Kill stuck process
Ctrl+C
```

---

## Week 1 Checklist

- [ ] Created accounts on all platforms (HackTheBox, TryHackMe, PortSwigger, CTFtime)
- [ ] Successfully installed and configured Kali Linux
- [ ] Completed basic Linux commands practice
- [ ] Written and executed first bash scripts
- [ ] Completed TryHackMe Linux Fundamentals rooms
- [ ] Finished all three scripting assignments
- [ ] Explored each cybersecurity platform
- [ ] Prepared for Week 1 CTF challenge

**Next Week Preview:** Cryptography, Steganography & Basic Forensics - where we'll dive into the fascinating world of hidden messages and digital evidence!

---

*Remember: Cybersecurity is a journey, not a destination. Every expert was once a beginner. Don't hesitate to ask questions and help your fellow students!*
