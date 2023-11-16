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
- mcelog
  - Logs and accounts machine checks (memory, IO, CPU error) on modern x86 Linux systems
  - /var/log/mcelog or syslog or the journal
- rasdaemon
  - Replacement of mcelog
  - systemctl start rasdaemon
  - ras-mc-ctl --status
    - Will not work on VM. Only baremetal (?)

20. Memory testing using memtest86+ package
- yum install memtest86+
- memtest-setup
  - copy paste the command from message
  - grub2-mkconfig -o /boot/grub2/grub.cfg
  - Will restart and do the memtest

21. Managing Kernel Modules
- Linux kernel has large parts of its functionality split out into modules, small pieces of code that can be loaded and unloaded at will. This helps keep the base kernel image smaller, so that only code that is actually needed is loaded in memory
- lsmod  # all modules
- /sys/module # all modules 
- modinfo <module_name>

22. Module options

23. Practice Lab session

## Section 4: Troubleshooting Storage Issues

24. Linux Storage Stack
- How IO is passed from application to disk
  - An application calls a system call to kernel
  - Kernel responds through storage stack
    - storage stack: various layers of SW/HW for the access to storage
    - vfs (virtual file system) 
      - xfs
      - ext3
      - ext4
      - page cache
      - device mapper

25. Virtual File System (VFS)
- Application -> Virtual File system -> file system -> Block FS or Network FS -> Block IO layer -> physical storage devices
- The Virtual File System (VFS)
  - Standard system calls: read(), open(), write()
  - Common file model
  - Various IO cache
    - inodes
    - dentries
    - Buffer cache: uses disk block
    - Page cache: uses main memory
- How to check buffer/cache size
  - cat /proc/meminfo
  - free -m
- /proc/sys/vm/drop_caches
  - Ref: https://www.kernel.org/doc/Documentation/sysctl/vm.txt
```
Writing to this will cause the kernel to drop clean caches, as well as reclaimable slab objects like dentries and inodes.  Once dropped, theirmemory becomes free.
To free pagecache:
	echo 1 > /proc/sys/vm/drop_caches
To free reclaimable slab objects (includes dentries and inodes):
	echo 2 > /proc/sys/vm/drop_caches
To free slab objects and pagecache:
	echo 3 > /proc/sys/vm/drop_caches
```
  - Dropping caches after slum job completion: https://lists.openhpc.community/g/users/topic/drop_caches_in_slurm_epilog/92634068?p=,,,20,0,0,0::recentpostdate/sticky,,,20,0,20,92634068,previd%3D1660875192566568171,nextid%3D1655315606927954914&previd=1660875192566568171&nextid=1655315606927954914
  - Add `echo 3 > /proc/sys/vm/drop_caches` into /etc/slurm/slurm.epilog.clean script

26. File system and Device mapper
- Device mapper: create 1:1 mapping of blocks in one block device to blocks in another, logical block device
- dmsetup ls # check device
- fdisk /dev/sdb # Then enter p to see the detail of /dev/sdb. Enter q to exit. We split /dev/sdb into 2 pieces
- pvcreate /dev/sdb1 /dev/sdb2 # creation of physical volume
- vgcreate /dev/vgtest /dev/sdb1 /dev/sdb2 # creation of volume group
- lvcreate -L +20M -n lvol1 /dev/vgtest # creation of logical volume
- ls -l /dev/mapper/vtest-lvol1
- lvcreate -L 140M -n lvol2 /dev/vgtest
- ls l /dev/mapper/vgtest-lvol2
- lvdisplay -v /dev/mapper/vgtest-lvol2
- dmsetup table /dev/mapper/vgtest-lvol2

27. DM Multipathing
- Device Mapper Multipath: a Linux native multipath tool, which allows to configure mutiple IO paths b/w server nodes and storage arrays into a single device
- When a server connects to storage through multiple SAN connections
  - Can use only one interface
  - Can increase throughput
- yum -y install device-mapper-multipath
- cp /usr/share/doc/device-mapper-multipath-X.X.X/multipath.conf /etc
- lsmod |grep -i dm_multipath
- systemctl start multipathd
- systemctl enable multipathd

28. Recovering File System Corruption
- File system selection
  - ext2, ext3, exr4, xfs
- Identifying file system corruption
  - Check logs

29. Checking & Repairing ext3/ext4 fs using e2fsck
- e2fsck : will fix corrupted disk
  - e2fsck -n /dev/sdb1 # file system check. -n means answering No to all queries
- Sample run  
  - umount /test # find dev from df -h
  - e2fsck -n /dev/sdb1
  - dumpe2fs /dev/sdb1 # can locate the Backup superblock
  - e2fsck -y /dev/sdb -b XXX # block location from dumpe2fs
  - dd if=/dev/zero of=/dev/sdb1 bs=512 count 4 # artificially corrupting disk  
  - mount /dev/sdb1 /test # will not work as the disk is corrupted
  - e2fsck /dev/sdb1 # enter y when queried

30. Checking & Repairing xfs fs using xfs_repair
- mkfs -t xfs -f /dev/sdb1
- mkdir /xfs_fs
- mount /dev/sdb1 /xfs_fs/
- sudo blkid # shows file systems
- xfs_repair -n /dev/sdb1 # -n for No to queries
  - Will not work as mounted already
- umount /xfs_fs; xfs_repair -n /dev/sdb1
- Let's corrupt little
  - dd if=/dev/zero of=/dev/sdb1 bs=1024 count=2
  - mount /dev/sdb1 /xfs_fs/ # will fail as corrupted
  - xfs_repair -n /dev/sdb1 # Says the primary superblock is bad
  - xfs_repair /dev/sdb1; mount /dev/sdb1 /xfs_fs/ # now works

31. Recovering LVM Issues
- fdisk /dev/sdb
  - p to print the status
  - d to delete
  - n to create
    - p for primary
  - t to change type of partition
    - 8e for Hex code
  - w to save and exit
- pvcreate /dev/sdb1
- vgcreate /dev/vgtest /dev/sdb1
- /etc/lvm/backup
- lvcreate -L +100M /dev/vgtest
- Check /etc/lvm/backup/vgtest
- lvcreate -L 100M -n lvol10 /dev/vgtest # creating a new logical volume with the name lvol10
  - Check /etc/lvm/archive/vgtest folder and find a new *.vg file
- lvs # list of logical volumes
- lvremove /dev/vgtest/lvl10 # removes a logical volume

32. Practice Lab Session - using vgcfgrestore
- How to revert LVM creation
- vgcfgrestore -l vgtest # volume group configuration restore command
- vgcfgrestore -f /etc/lvm/archive/vgtest_XXXX-XXXX.vg vgtest # find the *.vg file from -l command above
  - Answer y to restore
  - lvchange -ay /dev/vgtest/lvol10
  - lvdisplay /dev/vgtest/lvol10

33. Migrating a Volume Group from one system to another

34. Recovering Metadata in LVM
- Critical information in the header
- Symptoms
  - lvs -a -o +devices
  - Couldn't find device with uuid XXXXX
  - Couldn't find all physical volume XXXX

35. Practice Lab Session
- Among vg_snap, we corrupt /dev/sdg1 manually
- dd if=/dev/zero of=/dev/sdg1 bs=1024 count=1
  - Artificially deletes the header
- vgdisplay -v vg_snap # label of a volume group
  - /dev/sdg1 will show unknown device as the header is corrupted
- lvs -a -o +devices # generates WARNING that device for PV XXXX not found
- vgchange -an --partial # inactivate volume group except opened volume (?)
- pvcreate --uid "XXXX" --restorefile /etc/lmv/backup/vg_snap  /dev/sdg1  # using the same id of the corrupted vg
- vgcfgrestore /dev/vg_snap
- vgchange -a y /dev/vg_snap
- vgdisplay -v vg_snap
- Open /etc/lvm/backup/vg_snap and look for disks

36. Configuration of lscsi target & Initiator
- Class for iSCSI is missing. Following is from the enclosed ppt:
  - Internet Small Computer System Interface (iSCSI) is an TCP/IP-based standard for connecting storage devices. iSCSI uses IP networks to encapsulate SCSI commands, allowing data to be transferred over long distances.
  - iSCSI provides shared storage among a number of client systems. Storage devices are attached to servers (targets). Client systems (initiators) access the remote storage devices over IP networks. 
  - To the client systems, the storage devices appear to be locally attached. iSCSI uses the existing IP infrastructure and does not require any additional cabling, as is the case with Fibre Channel (FC) storage area networks.
- yum install targetcli
- ll /usr/lib/systemd/system/target.service
- targetcli
  - Use ls command
  - Create backing storage (backstores)
    - block: A block device such as disk drive, disk partition, logical volume, ...
    - fileio: Creates a file of specified size
    - pscsi: Physical SCSI. Permits passthrough to a physical SCSI device
    - ramdisk: creates a ramdisk of specified size in memory on the server

37. Practice lab session on iscsi target & initiator - 1 
- yum install iscsi-init*
- /etc/iscsi/initiatorname.iscsi

38. Practice lab session on iscsi target & initiator - 2

## Section 5: Troubleshooting RPM Issues

39. Resolve package dependencies
- rpm does not resolve dependency issues
- yum resolves dependencies
  - /var/log/yum.log contains a history of installed and erased packages
  - /var/log/apt/history.log or term.log for Ubuntu
  - yum deplist XXXX # list of dependency
- rpm -q --requires XXX # list of dependency from rpm

40. Identify & fix dependency issue
- yum -y downgrade XXX # downgrade the package
- rpm -qa | grpe -i XXX # check the installed rpm
- yum list XXX # check XXX availability from repo
- yum versionlock add XXX # when the update of XXX must be avoided from yum install/update
  - yum install XXX # will not update the package
- yum versionlock delete XXX # deletes the version lock for XXX

41. Recover a corrupted RPM database
- When `rpm -qa` not working
  - lsof /var/lib/rpm
  - Make backup of /var/lib/rpm
  - cd /var/lib/rpm 
  - ll ./Packages
  - ll /usr/lib/rpm/rpmdb_verify
  - /usr/lib/rpm/rpmdb_verify ./Packages
    - May take some time
    - Verification succeeded or failed
    - When fail,  /usr/lib/rpm/rpmdb_load ./Packages
      - Will create a new file
      - Run rpmdb_verify again to make sure
- rpm -v --rebuilddb # rebuild all indexes

42. Working with transaction history using yum command
- yum history list
- yum history list all
- yum history info <id>
  - <id> number from history list
- yum history sync
- In Ubuntu, check /var/log/apt/history*

43. Reverting and repeating transaction using yum command
- yum history undo <id>
- yum history redo <id>
- yum history new

## Section 6: Troubleshooting Network Issues

44. Check Network Connectivity
- ping (icmp protocol)
  - ping www.cnn.com
  - ping -i 3 www.cnn.com # every 3sec
  - ping -I enp3s0 www.cnn.com # using a different network interface
    - Find interface from ifconfig
  - ping -c 5 www.cnn.com # 5 times of request only
  - ping -W 3 www.cnn.com # 3 sec for timeout
    - If the IP responds, it will not timeout

45. Scanning network ports using nmap
- yum install nmap
- nmap my.workstation.com # home-network only
  - Will scan which ports are open
  - May need permission to scan: https://nmap.org/book/legal-issues.html
    - Do not scan public IP

46. Communicating with a remote service using nc command
- NCAT: a tool to communicate with a service. Can use either of TCP and UDP while supports SSL as well. Line command is `nc`
- nc workstation.example.com 25

47. Network Monitoring Utility - iptraf-ng
- Real time monitoring of IP traffic

48. Troubleshooting network issue - 1
- Scenario
  - One network interface works while the other not
  - ifconfig # find available interfaces
  - nmcli dev status
  - ip addr
  - nmcli conn
  - Check /etc/sysconfig/network-scripts/ifcfg-XXX
    - The name of interface (XXX) must meet the ones found from nmcli dev status
    - Check if name and device name is correct
  - Now reload : nmcli con reload; nmcli dev status

49. Troubleshooting tips related with device names of ethernet interface
- eno1: en for ethernet, o for on-board, then device number
- ens1: en for ethernet, s for slot
- /etc/udev/rules.d/80-net-name-slot.rules for persistent naming
  - Can make customized rule as well

50. Disable consistent network device naming

51. Overview NetworkManager
- NetworkManager
  - A daemon monitors and manages network settings
  - Managed using nmcli and other GUI tool
  - /etc/sysconfig/network-scripts is updated in the back-end
    - /etc/NetworkManager/system-connections in Ubuntu
```bash
$ nmcli dev status
DEVICE           TYPE      STATE        CONNECTION      
wlp4s0           wifi      connected    Bucs            
enp3s0           ethernet  unavailable  --              
lo               loopback  unmanaged    --    
```          
  
52. Practice Lab Session - 1

53. Practice Lab Session - 2

54. Capturing packets with tcpdump
- When a server responds slowly, may be used to investigate the packets

55. Practice Lab Session - 3
- tcpdump -i eth0 # using an interface eth0
- tcpdump -c 5 -i eth0 # 5 counts
- tcpdump -n -c 5 -i eth0 # better description
- tcpdump -n -c 5 -i eth0 port 22 # port 22 only
- tcpdump -w 0001.pcap -i eth0 # write to 0000.pcap
 
## Section 7: Troubleshooting Boot Issues

56. A traditional boot sequence
- As Power up
  - system startup by Bios/Bootmonitor
  - Stage 1 bootloader by Master Boot Record
  - Stage 2 bootloader by LILO, GRUB, ...
  - Kernel by Linux
  - Init by User space
    - /sbin/init
    - /etc/rc.d/rc.sysinit
    - /etc/inittab

- MBR
  - 446 bytes of bootloader
  - 64 bytes of partition table
  - 2 bytes of magic number

57. targets/runlevels in boot process
- Run level and scripts directory
  - /etc/rc.d/rc0.d/ : Shutdown/hal system
  - /etc/rc.d/rc1.d/ : Single user mode
  - /etc/rc.d/rc2.d/ : Multiuser with no network services
  - /etc/rc.d/rc3.d/ : Default txt/console only. Full multiuser
  - /etc/rc.d/rc4.d/ : Reserved for local use. X-windows (slackware/BSD)
  - /etc/rc.d/rc5.d/ : XDM X-windows GUI (Redhat/System V)
  - /etc/rc.d/rc6.d/ : Reboot
  - s or S # single user/maintenance mode (slackware)
  - M : Multiuser mode (slackware)
  - Each folder has K* or S* scripts
    - K for kill process
    - S for start

58. Overview of Grub2
- GRand Unified Bootloader: GRUB
- Mediator b/w BIOS and OS Kernel
- Allows multiple kernels and users can select one at boot time

59. Configuring grub2
- grub.cfg: made through grub2-mkconfig
  - /boot/grub2/grub.cfg is generated at Linux installation and regenerated when a new kernel is installed
- Kernel index
  - 0: current kernel
  - 1~...: old kernels
- /etc/grub.d folder has 00_header, 10_linux, ... files
  - At /boot/grub2/grub.cfg
```
### BEGIN /etc/grub.d/00_header ###
if [ -s $prefix/grubenv ]; then
...
### END /etc/grub.d/00_header ###
```  
- Keep the backup of /boot/grub2/grub.cfg

60. Grub2 features
- GRUB2 menu configuration settings are taken from /etc/default/grub
  - After edit, run grub2-mkconfg then new grub.cfg will be produced - `grub2-mkconfig -o /boot/grub2/grub.cfg`
- grub2 searches the compressed kernel image file called vmlinz in /boot
- grub2 load the vmlinuz into memroy and extracts the content of the initramfs image file into a temporary, memory based file system (tmpfs)
- The initial RAM disk (initrd) is an initial root file system that is mounted before the real root file system

61. Booting into grub menu
- At grub menu -> e key to edit
- At grub menu -> c to get command shell
  - linux16 /vmlinuz-3.10.0-327.el7.x86)64 root=/dev/sda2
  - init
  - initrd16 /init
  - initrd16 /initramfs-3.10.0-327.el7.x86_64.img
  - boot
  - Now will boot using the img

62. Protect grub by applying passwd
- grub-md5-crypt
  - Get hashed passwd
- Then edit /boot/grub/grub.conf, injecting `password --md5 XXXXXX`
- In grub2:
  - Edit /etc/grub.d/10_linux and remove `--unrestricted` in CLASS definition
  - Run grub2-setpassword
  - Will generate /boot/grub2/user.cfg
  - grub2-mkconfig -o /boot/grub2/grub.cfg

63. initramfs file missing or corrupted
- Screen will be blank at boot
- Load iso image from optical or usb
- Enter rescue mode
- Start shell prompt
- chroot /mnt/sysimage
- Recreate initramfs
  - cd /boot
  - `dracut -f initramfs-$(uname -r).img $(uname-r)`
- Or use grub console
  - root (hd0,0)
  - setup(hd0)
- Reboot

64. grub file missing
- When grub conf file is missing
- At boot, grub prompt
- root (hd0,0)
  - For different partitions, root (hd0,1)
- kernel /vmlinuz-2.32-220.el6.x86_64 ro root=/dev/mapper/vg_sat_lv # make sure to use the correct name
- initrd /initramfs-2.6.32-220.el6.x86_64.img
- Login to the system and recover grub.cfg

65. mbr corrupted - 1
- Deleting MBR
  - df -h # find the dev like /dev/sda1
  - dd if=/dev/zero of=/dev/sda bs=256 count=1 # 1st 256 block of /dev/sda will be zeroed
- reboot
  - As MBR is gone, it will find next bootable device
  - Or mount CD or usb if there is no available device
    - Enter Rescue mode

66. mbr corrupted - 2
- Recovering MBR
  - chroot /mnt/sysimage
  - df -h # check mounted system
  - grub-install /dev/sda # bootable device
  - Reboot
- Or run grub prompt
  - root (hd0,0) # select the first partition
  - setup (hd0)
  - quit # then reboot

67. Rescue mode in RHEL7
- Single user mode with root passwd
- Will mount all local file systems
- No active network/no multiple users
- Activating rescue mode
  - Select rescue mode from grub menu
  - Or select `e` from grub menu, then append `systemd.unit=rescue.target` to the linxu16 command
    - ctrl+a for the start of the line or ctrl+e for the end of the line
    - ctrl+x to boot

68. Practice Lab Session

69. Reset the root passwd using installation disk
- Troubleshooting mode from bootable disk
  - Choose Rescue mode
  - Choose Continue
  - chroot /mnt/sysimage
  - passwd root # Now can reset root passwd

70. Reset the root passwd using rd.break
- Edit mode from grub boot screen
- Remove rhgb and quiet parameters from linux16 line command or linuxefi
- Append `rd.break enforcing=0` to the linux16 or linuxefi command
- ctrl+x to boot
- If a prompt is not obvious, press **Backspace** 
- File system is mounted as read-only on /sysroot
- mount -o remount,rw /sysroot # remount as writable
- chroot /sysroot
- df -h # will not work
- passwd # can change passwd

71. Repairing File systems issues at boot
- systemd attempts to repair bad file system but emergency shell may appear if it is too severe
- Non-existent device or UUID at /etc/fstab
- Non-existent mount point in /etc/fstab
- In-correct mount options in /etc/fstab
- Can add `nofail` option in /etc/fstab to avoid failure
  - Not recommended to use
  - Application may not notice the absence of the device

72. Repair file systems issues - lab
- When /etc/fstab is corrupted
  - Boot message shows failure at mounting
  - Will ask root passwd for maintenance
  - mount # shows mounted systems
    - Find any read-only mounted systems
  - mount -o remount.rw / # remount all system as writable
  - Edit /etc/fstab

73. Repair file systems issues - lab
- When /etc/fstab is corrupted
  - Boot as rescue mode
  - mount # find any read-only systems
  - mount -o remount.rw /dev/mapper/rhcl-root /
  - Edit /etc/fstab
  - systemctl daemon-reload
  - exit

## Section 8: Troubleshooting Security Issues

74. Troubleshooting a SELinux issue
- getenforce
- chcon -R 

75. Changing SELinux context
- unconfined_u
- object_r
- default_t
- chcon -t httpd_sys_content_t /testselinux
- restorecon -v /testselinux
- semanage fcontext -a -t httpd_sys_content_t /testselinux
- ls -ltrZ /testselinux

76. Troubleshooting ftp connectivity issue using booleans
- getenforce
- ps -efZ|grep -i vsftpd
- cd /var/ftp
- ls -ltrdZ /var/ftp
- getsebool -a |grep -i ftp_home
- setsebool -P ftp_home_dir on

77. SELinux Audit Logs & Troubleshooting
- cd /var/log
- ll messages
- cd audit
- Check audit.log
- Or use SELinux Alert Browser

78. Overview of PAM security
- Pluggable Authentication Modules
  - Management layer b/w Linux applications and the Linux native authentication system
  - /etc/passwd and /etc/shadow
  - Users like www, apache, ftp, ...

79. Concepts of PAM
- /etc/pam.d and /etc/security

80. PAM Modules & Configurations
- PAM modules
  - /lib/security
  - /lib64/security
  - pam_tally2
  - pam_unix
  - pam_localuser 
- PAM configuration
  - /etc/pam.d
  - /etc/security
  - /etc/pam.d/vsftpd
  - /etc/pam.d/sshd

81. PAM Module Groups
- PAM configuration file
  - Module_Interface
    - Auth group
    - Account group
    - Session group
    - Password group
  - Control_flag
  - Module_Name
  - Module_Arg

82. Control Flags in PAM
- Order matters
```
auth requisite pam_securetty.so
auth required pam_rootok.so
auth sufficient pam_ldap.so
```

83. PAM Modules
- pam_unix Module
  - Checks user's credentials against /etc/shadow
  - auth requires pam_unix.so
  - auth, session, password management group
- pam_deny Module
  - Restricts access. 
  - Use this module in the end of modle to protect from misconfiguration
  - Use this module in the beginning then service will be disabled
  
84. Last Lecture
