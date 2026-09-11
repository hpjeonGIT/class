# Linux and X11
- Summary and note for operating/architecting Linux and X11 over Linux workstation/cluster
  - Updated as of Sept 2026
- X11 will be replaced with Wayland

## Systemd comand line utilities
- systemctl
- journalctl
- systemd-analyze
- systemd-run
- loginctl
- networkctl # Not found in RHEL package?
- timedatectl
- localectl
- hostnamectl
- coredumpctl
- resolvectl
 
## Run level: mode of operation that implements Unix System V-style initialization
- Ref: https://www.cyberciti.biz/tips/linux-changing-run-levels.html
- Each runlevel is configured at /etc/rc0.d, rc1.d, rc2.d, rc3.d, rc4.d, rc5.d, rc6.d
  - Each symbolic link starting with 'S*' means starting process
  - Each symbolic link starting with 'K*' means killing process
- Runlevel 0
  - `sudo init 0` # shutdown/halt
  - Systemd targets: runlevel0.target, poweroff.target
- Runlevel 1
  -`sudo init 1` # single user mode
  - Systemd targets: runlevel1.target, rescue.target
- Runlevel 2
  - `sudo init 2` # multi user without networking
  - Systemd targets: runlevel2.target, multi-user.target
- Runlevel 3
  - `sudo init 3` # multi user with networking. an alias of `sudo systemctl isolate multi-user.target`
  - Systemd targets: runlevel3.target, multi-user.target
- Runlevel 4 # user-definable
  - Systemd targets: runlevel4.target, multi-user.target
  - https://www.thegeekyway.com/runlevels-and-rc-d-scripts/
- Runlevel 5 # multiple user mode undere GUI - standard for most Linux-based system
  - Systemd targets: runlevel5.target, graphical.target
- Runlevel 6 # reboot
  - Systemd targets: runlevel6.target, reboot.target
- Related command:
```bash
$ runlevel
N 5
$ who -r
         run-level 5  2026-08-13 16:23  
$ systemctl get-default
graphical.target
```
- Installing CUDA driver after OS kernel update
  - `sudo yum update -y` # including kernel update
  - `sudo reboot`
  - When reboots, GUI will be broken
  - ssh to the server/workstation or ctrl+alt+backspace or ctrl+alt+F2 for CLI login
  - `sudo init 3` # Disable GUI
  - `sudo sh ./Nvidia-driver-XXXX.sh` # Re-install CUDA driver

## Boot target
- Types
  - graphical.target
  - multi-user.target
  - rescue.target
  - emergency.target
  - poweroff.target
  - reboot.target
- layered targets: sysinit.target -> basic.target -> multi-user.target -> graphical.target
- checking the current default target: systemctl get-default
- set to no GUI: `sudo systemctl set-default multi-user.target` # server mode
- enable GUI: `sudo systemctl set-default graphical.target`     # most of workstations
- without rebooting:
    - `sudo systemctl isolate multi-user.target`
    - `sudo systemctl isolate graphical.target`

## Nitty-gritty on X11
- Ref: https://en.wikibooks.org/wiki/Guide_to_X11/Starting_Sessions
- xinit: X window system initializer
- startx
  - Initializes an X session
  - A shell script running xinit
  - May use $HOME/.xinitrc or /etc/X11/xinit/xinitrc or /etc/X11/xinit/xserverrc
- xwd: dumps an image of an X window
- Relative system variables
```bash
$ echo $XAUTHORITY
/run/user/1000/gdm/Xauthority
$ echo $DISPLAY
:1
```
- What is seat0?
  - Ref: https://wiki.archlinux.org/title/Xorg_multiseat
```bash
$ loginctl
SESSION  UID USER   SEAT  TTY  STATE  IDLE SINCE
      2 1000 hpjeon seat0 tty2 active no   -    

1 sessions listed.
$ loginctl list-seats
SEAT 
seat0

1 seats listed.
$ loginctl seat-status seat0
seat0
Sessions: *2
 Devices: n/a
         ├─/sys/devices/pci0000:00/0000:00:02.0/drm/card1
         │ [MASTER] drm:card1
         │ ├─/sys/devices/pci0000:00/0000:00:02.0/drm/card1/card1-DP-1
         │ │ [MASTER] drm:card1-DP-1
...         
         ├─/sys/devices/virtual/input/input17
         │ input:input17 "Veikk Stylus"
         ├─/sys/devices/virtual/input/input18
         │ input:input18 "Veikk Mouse"
         ├─/sys/devices/virtual/input/input19
         │ input:input19 "Veikk Keyboard"
         ├─/sys/devices/virtual/misc/kvm
         │ misc:kvm
         └─/sys/devices/virtual/misc/rfkill
           misc:rfkill
```
- `tmp/.X11-unix/X0`
  - Unix Domain Socket (UDS) for X11 display
  - https://unix.stackexchange.com/questions/196677/what-is-tmp-x11-unix
    - `man 7 socket`
    - https://www.geeksforgeeks.org/linux-unix/understanding-unix-sockets/ 
- How xauth works?
  - Manages authorization information using **magic cookie**
  - `xauth list`
- MIT-MAGIC-COOKIE-1
  - Ref: https://community.sap.com/t5/technology-blog-posts-by-members/use-of-magic-cookies-in-linux/ba-p/13880108
  - X11 authentication, creating a random secret value called a magic cookie
  - When cooki is correct, access is granted
  - The cookie is stored at ~/.Xauthority. Check by `echo $XAUTHORITY`
    - If ~/.Xauthority doesn't exist, /run/user/XXXX/gdm/Xauthority
  - How it works?
    - GDM or display manager creates a random cookie and stores at $Xauthority
    - when xterm or xlock runs, it reads cookie and sends it to the X server
    - If cookie is not accepted, `Invalid MIT-MAGIC-COOKEY-1 key` or `can't open display`
  - to see all cookies, `xauth list`
- Enabling GUI for root in a network-connected system
  - Adding MIT-MAGIC-COOKIE-1
    - `ssh -X serverA`
    - `xterm` # runs OK
    - `sudo xterm` # not working, saying wrong authentication
    - `xauth list` #
    - `xauth add hostname/unix:0 MIT-MAGIC-COOKIE-1 123456789`
    - `sudo xterm` # not working ? (Aug 2026)
  - Another approach
    - `echo $XAUTHORITY`
    - `sudo XAUTHORITY=/run/user/1929221304/gdm/Xauthority xterm` # this runs OK. Aug 2026 at Rocky8.10
  - Another method
    - `ssh -X serverA`
    - `echo $XAUTHORITY` # this is empty but '~/.Xauthority' is found
    - `sudo XAUTHORITY=/home/foo/.Xauthority xterm` works.
- what is /tmp/.X11-unix/X1024 ?    
  - Could be a left-over from XWayland
- XDG variables
  - XDG_DATA_DIRS
  - XDG_RUNTIME_DIR
  - XDG_SESSION_TYPE # x11 or wayland
  - XDG_CONFIG_DIRS  
- References:
  - https://medium.com/@bshreyasharma1/running-gui-applications-with-sudo-on-linux-by-sharing-xauth-cookies-48ea6e4c13fc
  - https://docs.citrix.com/en-us/linux-virtual-delivery-agent/current-release/configure/administration/others/xauthority.html
  - https://superuser.com/questions/1482471/xorg-x11-how-to-provide-cookie-based-access-to-x-server-using-xauth
  - https://packages.guix.gnu.org/packages/xauth/
  - https://goteleport.com/blog/_next/image/?url=%2Fblog%2F_next%2Fstatic%2Fmedia%2Fx11-forwarding-program.537e15a0.png&w=3840&q=75&dpl=31740931110
  - https://goteleport.com/blog/x11-forwarding?source=post_page-----70c239f02a2e---------------------------------------
  - https://medium.com/@nadzeya/ssh-x-forwarding-or-how-to-open-desktop-applications-on-linux-server-c198b00a6a55
  - https://medium.com/@saicoumar/configuring-x11-forwarding-over-ssh-1a5a21fc707b

## From x11 to Wayland
- Since ubuntu 25
- No more xterm or xeyes
- tigervnc not working anymore
- gnome-remote-desktop as rdp in MobaXterm?

## RDP for Rocky8 and mobaxterm
- Steps:
  - sudo dnf install xrdp             
  - sudo systemctl start xrdp
  - sudo systemctl enable xrdp
  - sudo firewall-cmd --add-port=3389/tcp --permanent
  - sudo firewall-cmd --reload
  - At mobaXterm, make a new RDP session - it conflicts with X11 at seat0. wayland doesn't support xrdp
  - xrdp vs gnome-remote-desktop
  - Mobaxterm doesn't support native wayland yet (Sept 2026)


### What is Active Directory?
- Active directory
  - by logical components
      1. Forest
      2. Tree
      3. Domain
      4. Organization Unit (OU)
      5. Objects
  - by physical components
      1. Domain contollers (DC)
      2. Global Catalog (GC)
      3. Sites
  - Core services
      1. LDAP: query directory info, search users/groups, update attributes
      2. Kerberos: authentication, single sign-on (SSO), ticket management
      3. DNS: locating domain controllers, locating kerberos services, service discovery
  - Administrative components
      1. Group Policies (GPOs)
      2. Security Groups
- How LDAP works with Active Directory

## Using Addr2line
- addr2line -e ./a.out -s <mem_addr> # find mem_addr from from objdump
- addr2line -e ./a.out -a <mem_addr>
- addr2line -e ./a.out -p <mem_addr>
- addr2line -e ./a.out --functions --demangle 0000000000401346
- addr2line -e ./a.out -f -C -p  0000000000401346
  - main at /home/ASC.LOCAL/bxj670/HW/cpp/time3.cpp:7
  - find memory address using `nm ./a.out |grep func_name` or backtrace within gdb
  - Look for the ip (Instruction Pointer) or pc (Program Counter) address.
 
## Avahi-daemon
- Zero-configuration networking like Bonjour in Apple
- Allows the machine to connect to devices using a `.local` name instead of remembering IP address
  - Instead of `ssh xx.yy.zz.qq`, `ssh serverB.local` works

## Debugging Format
- DWARF3, DWARF4, ...
- https://dwarfstd.org/dwarf4std.html
- 2014 LLVM Developer's meeting: "Debug Info Tutorial"
  - https://youtu.be/_wqX9H2A66M?si=f5OoFizjb1HvFoMF
  - What is DWARF then?
    - Standard for how to encode program structure, line, columns, variable
    - Permissive standard with vendor extensions
  - DWARF Vaguaries
     - DWARF consumers not generalized because they've only seen output from one tool (GDB works great with GCC's DWARF output)
     - is_stmt - a line table feature "indicating that the current instruction is a recommended breakpoint location" - omitting this caused GDB to do ... strange things
  - DWARF structure
    - debug_info: source construct description (functions, types, namespaces, etc)
    - debug_line: instruction -> source line mapping
    - Use llvm-dwarfdump and similar tools to examine this data in object and executable files
    - We skip the details of debug_loc, debug_ranges, debug_string, debug_abbrev, ...
  - DWARF structure - info
    - Hierarchical tag + attribute format
    - Something like a binary XML?
    - Tab_subprogram[], AT_name(), AT_decl_file(), AT_decl_line(), AT_type(), AT_external(), ...
  - DWARF structure - line table
    - State machine & all that
    - In order to save disk space
```
Addr Line File Flags
---- ---- ---- ---------
0x00     1    1 
0x07     2    1 prologue_end
0x8e     3    1 
0x18     3    1 end_sequence
```
  - LLVM uses DIBuilder for debug_info    
- How DWARF works: Debug information entries
  - https://calabro.io/dwarf/die  
