## Linux Diagnostics And Troubleshooting
- Instructor: Shikhar Verm

## Section 1: Introduction

1. Introduction

2. What is troubleshooting

3. Troubleshooting a login issue
- lastlog -u john # last login status
- getent passwd john # user information
  - If bash/csh is not defined, /bin/false is printed
- chsh -s /bin/bash john # Changing john's shell as bash

4. System information to aid in troubleshooting
- Collecting information
  - Using the system journal
    - systemd-journald: keeps the logs in memory
    - rsyslogd: gets messages sent to it by systemd-journald and sotres them on disk
  - Using Journalctl
    - Like systemctl, journalctl is also a systemd utility. For querying and displaying messages from the journal. Since the journal comprises of one or more binary files, journalctl is the standard way to read messages from it
    - Boot messages
      - journalctl -b : last booting message
      - journalctl -b -l
      - journalctl -b -2 : 2nd last boot
      - journalctl --list-boots
    - journalctl -ef : all of the logs
    - journalctl _SYSTEMD_UNIT=sshd.service
    - journalctl -p emerg..err : any emergency error message

5. Using the System Journal
- journalctl --since "2022-07-01 20:10:00" --until "2022-07-04" : to configure time range
- journalctl -o verbose : more verbosely
- Ubunut has journal file at /var/log/journal

6. Troubleshoot a web server issue using the log files
- elinks -dump http:/abc.example.com/abc.html
- Check /var/log/httpd/error_log or access_log
  - Check the permission of /var/www/html/abc.html
- ausearch -i -m avc -ts today
- chcon -t httpd_sys_content_t abc.html
- restorecon -Rv /var/www

7. Using RHEL Resources
- SOS report: A command in RHEL/CentOS, collecting system configuration and diagnostic information
- redhat-support-tool : CLI command from RHEL

8. Practice Lab Session - SoS report
- rpm -q sos
- yum -y install sos
- sosreport --help
- sosreport -k abc
  - Will generate /var/tmp/sosreport-....tar.xz

9. Practice Lab Session - FTP connectivity issue
- systemctl status vsftpd # check if it is active or not
- tail -f /var/log/messages # check any log messages
- ss -tulpn | grep -i ftp # check ports are open or not
- systemctl status firewalld # check firewall daemon
- firewall-cmd --list-all # check firewall port
- firewall-cmd --add-service=ftp --permanent # enables ftp
- firewall-cmd --reload
- ausearch -m avc -i -ts today
  - audit search command
- restorecon -Rv /var/ftp

## Section 2: Monitoring systems

10. System Monitoring with Cockpit
- cockpit: free/open source server admin tool that allows to easily monitor and configure
- sudo yum install cockpit
- sudo systemctl start cockpit
- sudo systemctl enable cockpit
- firewall-cmd --add-service=cockpit --permanent
- firewall-cmd --reload
- Open a web-browser with 9090 port
  - CPU/Disk/... status

11. Performance Co-pilot or pcp
- yum -y install pcp
- systemctl start pmcd
- systemctl start pmcd
- pmstat -s 5
  - Memory/swap/io/cpu status every 5 sec
- pminfo for keywords
- For historic data, use pmlogger
  - systemctl start pmlogger
  - cd /var/log/pcp/pmlogger then find the server name folder
    - Use pmval command to see the detail
- In Ubuntu:
```bash
$ sudo apt install pcp
$ pmstat
@ Sat Jul  9 19:46:21 2022
 loadavg                      memory      swap        io    system         cpu
   1 min   swpd   free   buff  cache   pi   po   bi   bo   in   cs  us  sy  id
    1.86 578156 350612 610600  3084m    1    0   12   67  259  787   2   1  98
    1.71 578156 350120 610600  3084m    0    0    0    6  340 1067   1   1  98
$ pminfo -dt kernel.cpu.util.user
kernel.cpu.util.user One-line Help: Error: One-line or help text is not available
    Data Type: double  InDom: PM_INDOM_NULL 0xffffffff
    Semantics: instant  Units: none
```

12. Centralized log server with Rsyslog
- /var/log
- On centos/RHEL, rsyslog daemon is the main log server preinstalled, followed by systemd journal daemon (journald)
- For clients, rsyslog can be sent to a server or be stored locally 
- Syntax of log message: type (facility).priority(severity) destination (where to send the log)
  - Facility: auth, cron, daemon, kernel, mail, syslog, lpr, local0-7, * for all
  - Severity: emerg(0), alert(1), err(3), warn(4), notice(5), info(6), debug(7), none

13. Practice Lab Session - 1
- systemctl is-active rsyslog
- ll /etc/rsyslog.conf # configuration file
  - TCP/UDP
  - Can add new logger

14. Practice Lab Session - 2
- Can alert each logger made above

15. Intrusion detection SW to monitor changes
- `aide`, Advanced Intrusion Detection Environment
  - Free open source intrusion detection tool
  - An independent static binary
  - Using predefined rules to check file and directory integrity
  - Creates a database from the regular expression rules defined in the configuration file
- /etc/aide/aide.conf
  - Can configure new rules
  - Can combine hash, rwx permission, SELinux, ...

16. Practice Lab Session - 3
- aide --init # creating database
  - Will scan all files in the configuration
  - At /var/lib/aide/aide.db.new.gz
    - Manually rename as /var/lib/aide/aide.db.gz (?)
- aide --check # compares with database
  - Reports any files added/removed/changed
- aide --update # Reflect the new change into database
  - Manually rename as /var/lib/aide/aide.db.gz 

17. System auditing with auditd
- Allows you to log and track access to files, directories, and resource of systems, as well as trace system calls
  - Can detect misbehavior or code malfunctions
- /var/log/audit/audit.log
- Adding new rules
  - auditctl -w /etc -p w -k etc_content # audits any files added/removed
  - auditctl -w /etc -p a -k etc_attribute # audits any permission change
  - Or edit /etc/audit/rules.d/audit.rules
- ausearch -i -k etc_content
- ausearch -i -k etc_attribute

## Section 3: Identifying Hardware issues

18. Identify hardware
- lscpu # identifies cpu
- lscpu -p # NUM status/cache sharing
- cat /proc/cpuinfo
- cat /proc/meminfo # memory info
- dmidecode -t memory # memory info
- lsscsi -v # detects disks
  - Use lsblk instead
- hdparm #get/set hard disk parameters
```bash 
$ sudo hdparm -I /dev/sda
/dev/sda:
ATA device, with non-removable media
	Model Number:       ST1000DM003-1ER162                      
	Serial Number:      Z4YAFCC3
	Firmware Revision:  CC45    
	Transport:          Serial, SATA 1.0a, SATA II Extensions, SATA Rev 2.5, SATA Rev 2.6, SATA Rev 3.0
...
```
- lspci # any devices connected to PCI
- lsusb # detects usb devices

19. Hardware error reporting using mcelog

20. Memory testing using memtest86+ package

21. Managing Kernel Modules

22. Module options

23. Practice Lab session

## Section 4: Troubleshooting Storage Issues

## Section 5: Troubleshooting RPM Issues

## Section 6: Troubleshooting Network Issues

## Section 7: Troubleshooting Boot Issues

## Section 8: Troubleshooting Security Issues
