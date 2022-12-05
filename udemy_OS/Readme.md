## Summary
- Title: Write Your Own Operating System From Scratch - Step by Step
- Instructor: x-BIT Development

## Section 1: Introduction

1. Introduction
- Requirements
  - X86 assembly
  - C language

2. MUST SEE
- A test computer
  - CPU
  - No HDD
  - PS/2 keyboard
- A USB flash drive
  - To boot
  - Use Rufus

3. OS concepts
- Address space
  - 64bit address
  - Byte addressing
    - 0, 1, 2, ... 2^64-1
- Process
  - Kernel (code, data)
  - Stack
  - Heap
  - Data
  - Instructions
- When multiple rocesses run, each process has Kernel-stack-heap-data-instructions structure each
- Operation modes
  - Kernel mode
    - Interface w/ HW
  - User mode
    - Where application runs

4. OS structures
- Monolithic system
  - System call + scheduling + Memory management + Device drivers at Kernel space
  - Direct communication among modules
  - An error in a module will panic entire OS
- Microkernel
  - OS is divided into small modules
  - Memory management+Device Drivers are at user space
  - Only Scheduling and Interprocess communication at Kernel space
  - Kernel is very small

## Section 2: Installation and Setup

5. Working with Windows10
- Windows subsystem for Linux
  - Run Ubuntu 
  - Install nasm
- Bochs x86 PC emulator
  - Make a `boot.img`
  - Enable 1GB page memory
  - ATA channel 9 20/16/63 for cylindeerrs/heads/Sectors per track

6. Working with MacOS

7. Working with Linux (Ubuntu)
- gcc, nasm
- `sudo apt install bochs`
- `sudo apt install bochs-x`
- bximage: 1-> hd -> flat -> 512 -> 10 -> boot.img
- bochs
  - Enable 1G page 
```bash
$ bximage  
========================================================================
                                bximage
                  Disk Image Creation Tool for Bochs
          $Id: bximage.c 11315 2012-08-05 18:13:38Z vruppert $
========================================================================

Do you want to create a floppy disk image or a hard disk image?
Please type hd or fd. [hd] 

What kind of image should I create?
Please type flat, sparse or growing. [flat] 

Enter the hard disk size in megabytes, between 1 and 8257535
[10] 

I will create a 'flat' hard disk image with
  cyl=20
  heads=16
  sectors per track=63
  total sectors=20160
  total size=9.84 megabytes

What should I name the image?
[c.img] boot.img

Writing: [] Done.

I wrote 10321920 bytes to boot.img.

The following line should appear in your bochsrc:
  ata0-master: type=disk, path="boot.img", mode=flat, cylinders=20, heads=16, spt=63
$ bochs
========================================================================
                        Bochs x86 Emulator 2.6
            Built from SVN snapshot on September 2nd, 2012
========================================================================
00000000000i[     ] LTDL_LIBRARY_PATH not set. using compile time default '/usr/lib/bochs/plugins'
00000000000i[     ] BXSHARE not set. using compile time default '/usr/share/bochs'
00000000000i[     ] lt_dlhandle is 0x55ad64847c70
00000000000i[PLGIN] loaded plugin libbx_unmapped.so
00000000000i[     ] lt_dlhandle is 0x55ad648492e0
00000000000i[PLGIN] loaded plugin libbx_biosdev.so
00000000000i[     ] lt_dlhandle is 0x55ad6484ac50
00000000000i[PLGIN] loaded plugin libbx_speaker.so
00000000000i[     ] lt_dlhandle is 0x55ad6484b470
00000000000i[PLGIN] loaded plugin libbx_extfpuirq.so
00000000000i[     ] lt_dlhandle is 0x55ad6484bc50
00000000000i[PLGIN] loaded plugin libbx_parallel.so
00000000000i[     ] lt_dlhandle is 0x55ad6484d890
00000000000i[PLGIN] loaded plugin libbx_serial.so
00000000000i[     ] lt_dlhandle is 0x55ad64851640
00000000000i[PLGIN] loaded plugin libbx_gameport.so
00000000000i[     ] lt_dlhandle is 0x55ad64851f00
00000000000i[PLGIN] loaded plugin libbx_iodebug.so
00000000000e[     ] Switching off quick start, because no configuration file was found.
------------------------------
Bochs Configuration: Main Menu
------------------------------

This is the Bochs Configuration Interface, where you can describe the
machine that you want to simulate.  Bochs has already searched for a
configuration file (typically called bochsrc.txt) and loaded it if it
could be found.  When you are satisfied with the configuration, go
ahead and start the simulation.

You can also start bochs with the -q option to skip these menus.

1. Restore factory default configuration
2. Read options from...
3. Edit options
4. Save options to...
5. Restore the Bochs state from...
6. Begin simulation
7. Quit now

Please choose one: [2] 3
------------------
Bochs Options Menu
------------------
0. Return to previous menu
1. Optional plugin control
2. Logfile options
3. Log options for all devices
4. Log options for individual devices
5. CPU options
6. CPUID options
7. Memory options
8. Clock & CMOS options
9. PCI options
10. Bochs Display & Interface options
11. Keyboard & Mouse options
12. Disk & Boot options
13. Serial / Parallel / USB options
14. Network card options
15. Sound card options
16. Other options
17. User-defined options

Please choose one: [0] 6

-------------
CPUID Options
-------------
0. Return to previous menu
1. CPUID vendor string: GenuineIntel
2. CPUID brand string:               Intel(R) Pentium(R) 4 CPU        
3. Stepping ID: 3
4. Model ID: 3
5. Family ID: 6
6. Support for MMX instruction set: yes
7. APIC configuration: xapic
8. Support for SSE instruction set: sse2
9. Support for AMD SSE4A instructions: no
10. Support for SYSENTER/SYSEXIT instructions: yes
11. Support for MOVBE instruction: no
12. Support for ADX instructions: no
13. Support for AES instruction set: no
14. Support for XSAVE extensions: no
15. Support for XSAVEOPT instruction: no
16. Support for AVX instruction set: 0
17. Support for AVX F16 convert instructions: no
18. Support for AVX FMA instructions: no
19. Support for BMI instructions: 0
20. Support for AMD XOP instructions: no
21. Support for AMD four operand FMA instructions: no
22. Support for AMD TBM instructions: no
23. x86-64 and long mode: yes
24. 1G pages support in long mode: no
25. PCID support in long mode: no
26. FS/GS BASE access instructions support: no
27. Supervisor Mode Execution Protection support: no
28. MONITOR/MWAIT instructions support: yes
29. Support for Intel VMX extensions emulation: 1

Please choose one: [0] 24

1G pages support in long mode? [no] yes

-------------
CPUID Options
-------------
0. Return to previous menu
1. CPUID vendor string: GenuineIntel
2. CPUID brand string:               Intel(R) Pentium(R) 4 CPU        
3. Stepping ID: 3
4. Model ID: 3
5. Family ID: 6
6. Support for MMX instruction set: yes
7. APIC configuration: xapic
8. Support for SSE instruction set: sse2
9. Support for AMD SSE4A instructions: no
10. Support for SYSENTER/SYSEXIT instructions: yes
11. Support for MOVBE instruction: no
12. Support for ADX instructions: no
13. Support for AES instruction set: no
14. Support for XSAVE extensions: no
15. Support for XSAVEOPT instruction: no
16. Support for AVX instruction set: 0
17. Support for AVX F16 convert instructions: no
18. Support for AVX FMA instructions: no
19. Support for BMI instructions: 0
20. Support for AMD XOP instructions: no
21. Support for AMD four operand FMA instructions: no
22. Support for AMD TBM instructions: no
23. x86-64 and long mode: yes
24. 1G pages support in long mode: yes
25. PCID support in long mode: no
26. FS/GS BASE access instructions support: no
27. Supervisor Mode Execution Protection support: no
28. MONITOR/MWAIT instructions support: yes
29. Support for Intel VMX extensions emulation: 1

Please choose one: [0] 
------------------
Bochs Options Menu
------------------
0. Return to previous menu
1. Optional plugin control
2. Logfile options
3. Log options for all devices
4. Log options for individual devices
5. CPU options
6. CPUID options
7. Memory options
8. Clock & CMOS options
9. PCI options
10. Bochs Display & Interface options
11. Keyboard & Mouse options
12. Disk & Boot options
13. Serial / Parallel / USB options
14. Network card options
15. Sound card options
16. Other options
17. User-defined options

Please choose one: [0] 7

--------------
Memory Options
--------------
0. Return to previous menu
1. Standard Options
2. Optional ROM Images
3. Optional RAM Images

Please choose one: [0] 1

----------------
Standard Options
----------------
1. RAM size options
2. BIOS ROM options
3. VGABIOS ROM options

Please choose one: [0] 1

----------------
RAM size options
----------------

Enter memory size (MB): [32] 1024

Enter host memory size (MB): [32] 1024

----------------
Standard Options
----------------
1. RAM size options
2. BIOS ROM options
3. VGABIOS ROM options

Please choose one: [0] 

--------------
Memory Options
--------------
0. Return to previous menu
1. Standard Options
2. Optional ROM Images
3. Optional RAM Images

Please choose one: [0] 
------------------
Bochs Options Menu
------------------
0. Return to previous menu
1. Optional plugin control
2. Logfile options
3. Log options for all devices
4. Log options for individual devices
5. CPU options
6. CPUID options
7. Memory options
8. Clock & CMOS options
9. PCI options
10. Bochs Display & Interface options
11. Keyboard & Mouse options
12. Disk & Boot options
13. Serial / Parallel / USB options
14. Network card options
15. Sound card options
16. Other options
17. User-defined options

Please choose one: [0] 12

------------------
Bochs Disk Options
------------------
0. Return to previous menu
1. First Floppy Drive
2. Second Floppy Drive
3. ATA channel 0
4. First HD/CD on channel 0
5. Second HD/CD on channel 0
6. ATA channel 1
7. First HD/CD on channel 1
8. Second HD/CD on channel 1
9. ATA channel 2
10. First HD/CD on channel 2 (disabled)
11. Second HD/CD on channel 2 (disabled)
12. ATA channel 3
13. First HD/CD on channel 3 (disabled)
14. Second HD/CD on channel 3 (disabled)
15. Boot Options

Please choose one: [0] 3

-------------
ATA channel 0
-------------

Channel is enabled: [yes] 

Enter new ioaddr1: [0x1f0] 

Enter new ioaddr2: [0x3f0] 

Enter new IRQ: [14] 

------------------
Bochs Disk Options
------------------
0. Return to previous menu
1. First Floppy Drive
2. Second Floppy Drive
3. ATA channel 0
4. First HD/CD on channel 0
5. Second HD/CD on channel 0
6. ATA channel 1
7. First HD/CD on channel 1
8. Second HD/CD on channel 1
9. ATA channel 2
10. First HD/CD on channel 2 (disabled)
11. Second HD/CD on channel 2 (disabled)
12. ATA channel 3
13. First HD/CD on channel 3 (disabled)
14. Second HD/CD on channel 3 (disabled)
15. Boot Options

Please choose one: [0] 4

------------------------
First HD/CD on channel 0
------------------------

Device is enabled: [no] yes

Enter type of ATA device, disk or cdrom: [disk] 

Enter new filename: [] boot.img

Enter mode of ATA device, (flat, concat, etc.): [flat] 

Enter number of cylinders: [0] 20

Enter number of heads: [0] 16

Enter number of sectors per track: [0] 63

Enter new model name: [Generic 1234]

Enter bios detection type: [auto]

Enter translation type: [auto]

------------------
Bochs Disk Options
------------------
0. Return to previous menu
1. First Floppy Drive
2. Second Floppy Drive
3. ATA channel 0
4. First HD/CD on channel 0
5. Second HD/CD on channel 0
6. ATA channel 1
7. First HD/CD on channel 1
8. Second HD/CD on channel 1
9. ATA channel 2
10. First HD/CD on channel 2 (disabled)
11. Second HD/CD on channel 2 (disabled)
12. ATA channel 3
13. First HD/CD on channel 3 (disabled)
14. Second HD/CD on channel 3 (disabled)
15. Boot Options

Please choose one: [0] 15

------------
Boot Options
------------
0. Return to previous menu
1. Boot drive #1: floppy
2. Boot drive #2: none
3. Boot drive #3: none
4. Skip Floppy Boot Signature Check: no
5. 32-bit OS Loader Hack

Please choose one: [0] 1

Boot from floppy drive, hard drive or cdrom ? [floppy] disk

------------
Boot Options
------------
0. Return to previous menu
1. Boot drive #1: disk
2. Boot drive #2: none
3. Boot drive #3: none
4. Skip Floppy Boot Signature Check: no
5. 32-bit OS Loader Hack

Please choose one: [0] 

------------------
Bochs Disk Options
------------------
0. Return to previous menu
1. First Floppy Drive
2. Second Floppy Drive
3. ATA channel 0
4. First HD/CD on channel 0
5. Second HD/CD on channel 0
6. ATA channel 1
7. First HD/CD on channel 1
8. Second HD/CD on channel 1
9. ATA channel 2
10. First HD/CD on channel 2 (disabled)
11. Second HD/CD on channel 2 (disabled)
12. ATA channel 3
13. First HD/CD on channel 3 (disabled)
14. Second HD/CD on channel 3 (disabled)
15. Boot Options

Please choose one: [0] 
------------------
Bochs Options Menu
------------------
0. Return to previous menu
1. Optional plugin control
2. Logfile options
3. Log options for all devices
4. Log options for individual devices
5. CPU options
6. CPUID options
7. Memory options
8. Clock & CMOS options
9. PCI options
10. Bochs Display & Interface options
11. Keyboard & Mouse options
12. Disk & Boot options
13. Serial / Parallel / USB options
14. Network card options
15. Sound card options
16. Other options
17. User-defined options

Please choose one: [0] 
------------------------------
Bochs Configuration: Main Menu
------------------------------

This is the Bochs Configuration Interface, where you can describe the
machine that you want to simulate.  Bochs has already searched for a
configuration file (typically called bochsrc.txt) and loaded it if it
could be found.  When you are satisfied with the configuration, go
ahead and start the simulation.

You can also start bochs with the -q option to skip these menus.

1. Restore factory default configuration
2. Read options from...
3. Edit options
4. Save options to...
5. Restore the Bochs state from...
6. Begin simulation
7. Quit now

Please choose one: [6] 4
Save configuration to what file?  To cancel, type 'none'.
[none] bochsrc
00000000000i[     ] write current configuration to bochsrc
Wrote configuration to 'bochsrc'.
------------------------------
Bochs Configuration: Main Menu
------------------------------

This is the Bochs Configuration Interface, where you can describe the
machine that you want to simulate.  Bochs has already searched for a
configuration file (typically called bochsrc.txt) and loaded it if it
could be found.  When you are satisfied with the configuration, go
ahead and start the simulation.

You can also start bochs with the -q option to skip these menus.

1. Restore factory default configuration
2. Read options from...
3. Edit options
4. Save options to...
5. Restore the Bochs state from...
6. Begin simulation
7. Quit now

Please choose one: [6] 7
00000000000i[CTRL ] quit_sim called with exit code 1
```
- Produced bochsrc:
```bash
# configuration file generated by Bochs
plugin_ctrl: unmapped=1, biosdev=1, speaker=1, extfpuirq=1, parallel=1, serial=1
, gameport=1, iodebug=1
config_interface: textconfig
display_library: x
memory: host=1024, guest=1024
romimage: file="/usr/share/bochs/BIOS-bochs-latest"
vgaromimage: file="/usr/share/bochs/VGABIOS-lgpl-latest"
boot: disk
floppy_bootsig_check: disabled=0
# no floppya
# no floppyb
ata0: enabled=1, ioaddr1=0x1f0, ioaddr2=0x3f0, irq=14
ata0-master: type=disk, mode=flat, translation=auto, path="boot.img", cylinders=
20, heads=16, spt=63, biosdetect=auto, model="Generic 1234"
ata1: enabled=1, ioaddr1=0x170, ioaddr2=0x370, irq=15
ata2: enabled=0
ata3: enabled=0
pci: enabled=1, chipset=i440fx
vga: extension=vbe, update_freq=5
cpu: count=1, ips=4000000, model=bx_generic, reset_on_triple_fault=1, cpuid_limi
t_winnt=0, ignore_bad_msrs=1, mwait_is_nop=0
cpuid: family=6, model=0x03, stepping=3, mmx=1, apic=xapic, sse=sse2, sse4a=0, s
ep=1, aes=0, xsave=0, xsaveopt=0, movbe=0, adx=0, smep=0, avx=0, avx_f16c=0, avx
_fma=0, bmi=0, xop=0, tbm=0, fma4=0, vmx=1, x86_64=1, 1g_pages=1, pcid=0, fsgsba
se=0, mwait=1
cpuid: vendor_string="GenuineIntel"
cpuid: brand_string="              Intel(R) Pentium(R) 4 CPU        "

print_timestamps: enabled=0
debugger_log: -
magic_break: enabled=0
port_e9_hack: enabled=0
private_colormap: enabled=0
clock: sync=none, time0=local, rtc_sync=0
# no cmosimage
# no loader
log: -
logprefix: %t%e%d
panic: action=ask
error: action=report
info: action=report
debug: action=ignore
keyboard: type=mf, serial_delay=250, paste_delay=100000, keymap=
user_shortcut: keys=none
mouse: enabled=0, type=ps2, toggle=ctrl+mbutton
parport1: enabled=1, file=""
parport2: enabled=0
com1: enabled=1, mode=null, dev=""
com2: enabled=0
com3: enabled=0
com4: enabled=0
```

8. How to use Resources

## Section 3: Boot Up

9. The first program
- BIOS
  - Basic Input Output System
  - Services run in real mode
  - Print characters
  - Disk services
  - Memory map
  - Video mode
- Boot
  - Find boot device and read the first sector from it into memroy location 0x7c00 (MBR code)
  - Jump to 0x7c00
- BIOS mode
  - Real Mode: boot process
  - Protected Mode: prepares for long mode
  - Long Mode: 64bit mode. Compatibility mode
    - Cannot print
- UEFI
- Real Mode
  - Load Kernel
  - Retrieve information about HW
  - Sgement Registeres (16bit)
    - cs : Code Segment
    - ds : Data Segment
    - es : Extra Segment
    - ss : Stack Segment
  - Address Format
    - Segment Register: offset(logical address)
    - Segment Register x 16 + Offset = Physical address
  - General purpose registers
    - 8bit: al ah bl bh
    - 16bit: ax bx cx dx
    - 32bit: eax ebx ecx edx
    - 64bit: rax rbx rcx rdx are not available in Real Mode
```assembly
[BITS 16]
[ORG 0x7c00]
start:
    xor ax,ax   
    mov ds,ax
    mov es,ax  
    mov ss,ax
    mov sp,0x7c00 ; 
PrintMessage:
    mov ah,0x13
    mov al,1
    mov bx,0xa ; 0xa means the character is printed in bright screen
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10  ; 
End:
    hlt    
    jmp End
Message:    db "Hello"
MessageLen: equ $-Message
times (0x1be-($-$$)) db 0
    db 80h      ; boot indicator
    db 0,2,0    ; starting CHS
    db 0f0h     ; type
    db 0ffh,0ffh,0ffh ; ending CHS
    dd 1        ; starting sector 
    dd (20*16*63-1)  ; size
    times (16*3) db 0
    db 0x55     ; signature
    db 0xaa	    ; signature
```
- Build script (build.sh)
```bash
nasm -f bin -o boot.bin boot.asm
dd if=boot.bin of=boot.img bs=512 count=1 conv=notrunc
```

10. Testing on Windoss 10

11. Testing on Linux (Ubuntu)
```bash
$ bash ./build.sh 
1+0 records in
1+0 records out
512 bytes copied, 0.000165192 s, 3.1 MB/s
$ bochs
...
<bochs:1> c # enter c to continue
```
![first_bochs](./bochs_11.png)
- "Hello" is printed
  - If `Message: dlopen failed for module 'x': file not found` is found at bochs, install bochs-x

- How to make a bootable USB to boot from a physical/actual computer:
```
$ sudo fdisk -l
$ sudo dd if=boot.img of=/dev/sdb bs=512 count=1
```

12. Testing on MacOS

13. Test Disk Extension Service
- Adding the check of disk extension
```assembly
[BITS 16]
[ORG 0x7c00]
start:
    xor ax,ax   
    mov ds,ax
    mov es,ax  
    mov ss,ax
    mov sp,0x7c00
TestDiskExtension:
    mov [DriveId],dl
    mov ah,0x41 ; ref=https://www.ctyme.com/intr/rb-0706.htm
    mov bx,0x55aa
    int 0x13
    jc NotSupport
    cmp bx,0xaa55
    jne NotSupport
PrintMessage:
    mov ah,0x13
    mov al,1
    mov bx,0xa
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10
NotSupport:
End:
    hlt    
    jmp End    
DriveId:    db 0
Message:    db "Disk extension is supported"
MessageLen: equ $-Message
times (0x1be-($-$$)) db 0
    db 80h
    db 0,2,0
    db 0f0h
    db 0ffh,0ffh,0ffh
    dd 1
    dd (20*16*63-1)	
    times (16*3) db 0
    db 0x55
    db 0xaa
```
- ./build.sh and bochs again

## Section 4: Loading the Loader and Switching to Long Mode

14. Loader
- Loader retrieves information about HW
- Prepares for 64-bit mode and switch to it
- Loader loads kernel in main memroy
- Jump to kernel
- Loader doesn't have 512 limit
- MBR code at 0x7c00
- Loader at 0x7e00
- Ref: https://www.ctyme.com/intr/rb-0708.htm
- boot.asm:
```assembly
[BITS 16]
[ORG 0x7c00]
start:
    xor ax,ax   
    mov ds,ax
    mov es,ax  
    mov ss,ax
    mov sp,0x7c00
TestDiskExtension:
    mov [DriveId],dl
    mov ah,0x41
    mov bx,0x55aa
    int 0x13
    jc NotSupport
    cmp bx,0xaa55
    jne NotSupport
LoadLoader:
    mov si,ReadPacket     ; offset field
    mov word[si],0x10     ; 0      size
    mov word[si+2],5      ; 2      number of sectors
    mov word[si+4],0x7e00 ; 4      offset
    mov word[si+6],0      ; 6      segment
    mov dword[si+8],1     ; 8      address lo
    mov dword[si+0xc],0   ; 12     address hi
    mov dl,[DriveId]
    mov ah,0x42
    int 0x13
    jc  ReadError
    mov dl,[DriveId]
    jmp 0x7e00 
ReadError:
NotSupport:
    mov ah,0x13
    mov al,1
    mov bx,0xa
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10
End:
    hlt    
    jmp End    
DriveId:    db 0
Message:    db "We have an error in boot process"
MessageLen: equ $-Message
ReadPacket: times 16 db 0
times (0x1be-($-$$)) db 0
    db 80h
    db 0,2,0
    db 0f0h
    db 0ffh,0ffh,0ffh
    dd 1
    dd (20*16*63-1)	
    times (16*3) db 0
    db 0x55
    db 0xaa
```
- loader.asm:
  - DriveId from boot.asm must be delivered to loader.asm
```assembly
[BITS 16]
[ORG 0x7e00]
start:
    mov ah,0x13
    mov al,1
    mov bx,0xa
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10    
End:
    hlt
    jmp End
Message:    db "loader starts"
MessageLen: equ $-Message
```
- build.sh
```bash
nasm -f bin -o boot.bin boot.asm
nasm -f bin -o loader.bin loader.asm
dd if=boot.bin of=boot.img bs=512 count=1 conv=notrunc
dd if=loader.bin of=boot.img bs=512 count=5 seek=1 conv=notrunc
```
- Build image
```bash
$ bash build.sh
1+0 records in
1+0 records out
512 bytes copied, 9.4243e-05 s, 5.4 MB/s
0+1 records in
0+1 records out
33 bytes copied, 7.9545e-05 s, 415 kB/s
```

15. Long Mode Support
- Checks if the CPU supports Long Mode
- loader.asm:
```assembly
[BITS 16]
[ORG 0x7e00]
start:
    mov [DriveId],dl
    mov eax,0x80000000
    cpuid
    cmp eax,0x80000001
    jb NotSupport
    mov eax,0x80000001
    cpuid
    test edx,(1<<29)
    jz NotSupport
    test edx,(1<<26)
    jz NotSupport
    mov ah,0x13
    mov al,1
    mov bx,0xa
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10
NotSupport:
End:
    hlt
    jmp End
DriveId:    db 0
Message:    db "long mode is supported"
MessageLen: equ $-Message
```

16. Load Kernel File
- Memory map
```
Free (reserved for HW)
--------- 0x100000
Reserved
--------- 0x80000
Kernel
--------- 0x10000
Free (reserved for HW)
Loader
--------- 0x7e00
Boot
--------- 0x7c00
Free
BIOS data Vectors
--------- 0
```
- loader.asm:
```assembly
[BITS 16]
[ORG 0x7e00]
start:
    mov [DriveId],dl
    mov eax,0x80000000
    cpuid
    cmp eax,0x80000001
    jb NotSupport
    mov eax,0x80000001
    cpuid
    test edx,(1<<29)
    jz NotSupport
    test edx,(1<<26)
    jz NotSupport
LoadKernel:
    mov si,ReadPacket
    mov word[si],0x10
    mov word[si+2],100
    mov word[si+4],0       ; offset
    mov word[si+6],0x1000  ; Segment 0x1000:0 = 0x1000*16+0 = 0x100000 Kernel memory map shown above
    mov dword[si+8],6
    mov dword[si+0xc],0
    mov dl,[DriveId]
    mov ah,0x42
    int 0x13
    jc  ReadError
    mov ah,0x13
    mov al,1
    mov bx,0xa
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10
ReadError:
NotSupport:
End:
    hlt
    jmp End
DriveId:    db 0
Message:    db "kernel is loaded"
MessageLen: equ $-Message
ReadPacket: times 16 db 0
```

17. Get Memory Map
- Ref: https://www.ctyme.com/intr/rb-1741.htm
- loader.asm:
```assembly
[BITS 16]
[ORG 0x7e00]
start:
    mov [DriveId],dl
    mov eax,0x80000000
    cpuid
    cmp eax,0x80000001
    jb NotSupport
    mov eax,0x80000001
    cpuid
    test edx,(1<<29)
    jz NotSupport
    test edx,(1<<26)
    jz NotSupport
LoadKernel:
    mov si,ReadPacket
    mov word[si],0x10
    mov word[si+2],100
    mov word[si+4],0
    mov word[si+6],0x1000
    mov dword[si+8],6
    mov dword[si+0xc],0
    mov dl,[DriveId]
    mov ah,0x42
    int 0x13
    jc  ReadError
GetMemInfoStart:
    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    mov edi,0x9000
    xor ebx,ebx
    int 0x15
    jc NotSupport
GetMemInfo:
    add edi,20
    mov eax,0xe820
    mov edx,0x534d4150
    mov ecx,20
    int 0x15
    jc GetMemDone
    test ebx,ebx
    jnz GetMemInfo
GetMemDone:
    mov ah,0x13
    mov al,1
    mov bx,0xa
    xor dx,dx
    mov bp,Message
    mov cx,MessageLen 
    int 0x10
ReadError:
NotSupport:
End:
    hlt
    jmp End
DriveId:    db 0
Message:    db "Get memory info done"
MessageLen: equ $-Message
ReadPacket: times 16 db 0
```
- We don't print the detailed memory map yet as print function is not implemented. Check Section 7 to print

18. Test A20 Line
- Ref: https://en.wikipedia.org/wiki/A20_line
   - For legacy BIOS boot loader
   - UEFI boot loaders use 32 bit protected mode or 64bit long mode

19. Set Video Mode

20. Protected Mode

21. Long Mode
