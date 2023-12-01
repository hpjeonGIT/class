## Udemy: Complete Linux Training with Troubleshooting Skills
- Instructor: Shikhar Verma 

## Section 12: Disk Partitions, fdisk & RAID configuration

82. GPT Partitioning scheme
- Unlike MBR, GPT has multiple partitioning table information

86. Introudction to RAID
- Redundant Array of Inexpensive disks
  - Now Redundant Array of independent disks
  - Combines multiple available disks into 1 or more logical drive and gives you the ability to survive one or more drive failures depending on the RAID level used

## Section 20: Kickstart Server - Automating installation

154. Overview of Kickstart Server
- Uses a text file to provide all of details and no interaction is required
- Kickstart in RHEL is similar to Jumpstart in Solaris

155. Features of Kickstart server
- Kicstart config file for all the details for OS installation
- Can install multiple servers in one go
- Can minimize manual intervention
- Requirement
  - NFS
  - HTTP
  - FTP

156. Theoretical steps to configure Kickstart Server
- Step 1: mount ISO and dump media source file
  - Using FTP for this demo but HTTP, HTTPS, NFS can be used as well
  - mount CD drive: `mount /dev/sr0 /mnt`
  - FTP server path: `cp -rfv /mnt/* /var/ftp/pub/`
- Step 2: Installation and generate Kicstart file
  - `yum install system-config-kickstart`
  - Generate kickstart config file by `system-config-kickstart`
    - GUI appears and you can configure:
      - Default language
      - Keyboard
      - Time zone
      - Root passwd
      - Target Architecture
      - Reboot system after installation
    
157. Graphical mode to configure Kickstart config file

158. Kicstart config file - 1 
- DHCP or static IP
- Firewall

159. Kicstart config file - 2

160. Lab 1

161. Lab2
- Using DHCP, ip will be assigned

## Section 38: Advanced Linux Commands

307. dig command
- DNS lookup utility
```bash
dig redhat.com
```

308. traceroute command
- Map the journey that a packet of information from source to destination
- As default, sends 3 packets and shows results of 3 cases
```bash
traceroute google.com
```

309. rsync
- -r : copies data recursively but doesn't preserve time stamp. use -t to keep to preserve time stamp

312. df, du & lsof
- df: disk free
- du: disk usage
- lsof: list of opened files
  - lsof -i TCP:1-1024

322. /proc
- proc file system is a virtual file system created on fly when system boots
- /proc/cpuinfo: cpu/mem info
- /proc/meminfo: memory info
- /proc/version: linux/kernel info
- /proc/partitions: paritions info

## Section 39: Introduction to Bash Shell

326. What is Linux Shell?
- Interface b/w users and Unix systems
 
## Section 46: Functions

392. Defining functions
```bash
$ hello() { echo "hello world"; }
$ hello
```
