## Title
- Learn Linux Kernel Programming by Linux Trainer
- Interfacing kernel modules using insmod/rmmod using printk, init, exit
- Actual kernel core programming is not addressed

## Section 1: Introduction

### 1. What is Device Driver

### 2. What is Kernel Module
- Kernal module: loadable. Not requiring rebooting of the system
- *.ko:kernel object
- /lib/modules show the list of kernels installed so far
- /lib/modules/`uname-r`/kernel
  - More than 5000 ko files

### 3. Device drivers vs Kernel modules
- Kernel module is not a device driver
- Advantage of kernel modules
  - can dynamically load necessary modules when necessary
  - no need of rebooting
- Disadvantage of kernel modules
  - consumes unpageable kernel memory
  - core functionality must be in the base kernel
  - security
- /boot/config-`uname -r` |grep CONFIG_MODULES
  - CONFIG_MODULES=y means that the system loads modules

### 4. Types of Kernel modules

### 5. Basic Commands
- lsmod : info of /sys/modules (in old days, /proc/modules)
- modinfo: info of the module. Shows the list of parameters or arguments of the module.

### 6. Hello World Kernel Module
- start/end for kernel module programming is done by modulle_init() and module_exit()
- Specify license using MODULE_LICENSE()
- include linux/module.h and linux/kernel.h. Use printk() for log level
- Sample code: hello.c
  - start returns 0 as int while end function is void
  - Needs Makefile from /lib/modules/`uname -r`/build/Makefile
  - build command: make -C /lib/modules/`uname -r`/build/ M=${PWD} modules
  - clean command: make -C /lib/modules/`uname -r`/build/ M=${PWD} clean

  - Makefile in the current path for hello.c
```
obj-m := hello.o
```
  - try: modinfo ./hello.ko
- insmod: insert module to the kernel
  - sudo insmod ./hello.ko ; lsmod |less; ls /sys/module/hello
- rmmod: remove module
  - sudo rmmod hello

### 7. printf vs printk
- No comma in printk()
- log_prirority: EMERG, ALERT, CRIT, ERR, WARNING, NOTICE, INFO, DEBUG

### 8. Simplified Makefile
- Update of Makefile
```
obj-m := hello.o
all:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean
```
- Make sure that the tab is injected in the first column

### 10. What happens if we return -1 from Kernel moduile init function
- Returning -1 from test_hello_init() will not be loaded in the kernel MODULE_LICENSE

### 11. Give another name to Kernel module
- Renaming as mymodule.ko
```
obj-m := mymodule.o
mymodule-objs := hello.o
all:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean
```

### 12. Kernel Module Span across Multiple C files
- use void for no-argument passing case
  - void func(void)
  - void func() may overload no list argument
- Multiple C source CONFIG_MODULES
```
obj-m := mymodule.o
mymodule-objs := hello.o func.o
all:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean
```

### 13. Two Kernel Modules from Single Makefile
- A single Make file and multiple modules
```
obj-m := hello1.o
obj-m += hello2.o
all:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean
```

### 14. Dmesg in deep
- dmesg command
  - dmesg -c or dmesg -C to clear the ring buffer
  - dmesg -l err,warn,crit -T # human readable, printing err,warn, critical messages
- /var/log/kern.log

### 15. Dmesg follow option
- Running dmesg as background
  - dmesg -w &
  - Can observe module print when loaded or unloaded in the CLI

### 16. Linux Kernel Module example with module_init only
- If the module source doesn't have module_exit(), rmmod will not work to remove the module
  - Reboot to remove the module

### 17. Linux Kernel Module example with module_exit only
- Without module_init(), having module_exit() only works for insmod and rmmod

## Section 2: Linux Kernel Module Internals

### 21. From .c to .ko

### 22. Understanding Module.symvers and modules.order
- hello.c -> hello.o and hello.mod.c -> hello.mod.o
  - hello.o and hello.mod.o -> hello.ko
- Module.symvers : external symbols defined in the module
- module.order : for multiple modules, it will list the order

### 23. Insmod vs Modprobe
- lsmod
- insmod : can load any modules
- modinfo
- rmmod
- modprobe : can load modules at /lib/modules/$(uname -r) only. Check and load dependency automatically

### 24. How Modprobe calculates dependencies (modules.dep/depmod)
- depmod : generates modules.dep and map files
- /lib/modules/`uname -r`/modules.dep

### 26. Examples of gcc attribute alias
- gcc alias. Check alias.c and alias_v.c

### 27. Linux Kernel Module example without module_init and module_exit macro
- Using init_module() and cleanup_module() instead of module_init() and module_exit()
- Check module_init_exit.c

## Section 3: Module Parameters

### 28. Passing parameters to Linux kernel modules
- Passing parameters: use module_param macro
- sudo insmod ./arguments.ko loop_count=5 name="hpjeon"
  - no space around `=` symbol
```
$ dmesg
[55604.584438] test_arguments_init: In init
[55604.584440] test_arguments_init: Loop Count:5
[55604.584441] test_arguments_init: Hi hpjeon
[55604.584442] test_arguments_init: Hi hpjeon
[55604.584442] test_arguments_init: Hi hpjeon
[55604.584443] test_arguments_init: Hi hpjeon
[55604.584443] test_arguments_init: Hi hpjeon
```
- Check /sys/module/arguments/parameters/loop_count and /sys/module/arguments/parameters/name

### 29. What happen if we pass incorrect values to module parameters
- Parameter data type will be checked.

### 30. How to pass parameters to builtin modules
- modprobe reads /etc/modprobe.conf for parameters (not existing for Ubuntu)

### 33. Passing array as module parameters
- Passing array as input
- sudo insmod ./parameter_array.ko  param_array=4,5,8

## Sectinon 4: Exportin Symbols

### 35. Symbol and Symbol Table
- symbol: name space for memory. Could be a variable or a function
- Every kernel image has a symbol table
  - sudo cat /boot/System.map-4.15.0-122-generic

### 36. Exporting symbol
- How to export a symbol
  - When you define a new function in the module
  - EXPORT_SYMBOL or EXPORT_SYMBOL_GPL

### 37. System.map vs /proc/kallsyms
- /boot/System.map vs /proc/kallsyms
  - /proc/kallsyms: contains symbols of dynamically loaded modules + built-in modules
  - /boot/System.map: only built-in modules

### 38. Linux Kernel Module example of exporting function
- sudo insmod symbol_export.ko
- sudo cat /boot/System.map-`uname -r` doesn't find print_jiffies
- sudo cat /proc/kallsyms  finds print_jiffies

### 39. Module Stacking
- Module stacking
  - New modules use the symbols exported by old modules
  - MSDOS file system relies on symbols from FAT module
- Steps for practice
  - sudo insmod ./module1.ko
  - sudo cat /proc/kallsyms |grep myadd
  - sudo insmod ./module2.ko
  - sudo ln -s <absolute_path>/module1.ko /lib/modules/`uname -r`/kernel/drivers/misc/
  - sudo ln -s <absolute_path>/module2.ko /lib/modules/`uname -r`/kernel/drivers/misc/
  - sudo depmod -a
    - Check /lib/modules/`uname -r`/modules.dep
  - sudo rmmod module2.ko
  - sudo rmmod module1.ko
  - sudo modprobe module2 # this works now
    - After adding dependency from modprobe, loading module2 will load module1 automatically

## Section 5: Module Licenses

### 43. What happens if we don't specify MODULE_LICENSE macro
- When MODULE_LICENSE is missing, verification fails and loading is disabled

### 44. What is tainted kernel
- Tainted kernel: not supported by community. Debugging functionality and API calls are limited.
  - Using proprietary kernel module may taint kernel

### 45. How to check whether the kernel is in tainted state or not
- How to check the kernel is tainted or not
  - Check dmseg
  - Check /proc/sys/kernel/tainted. When larger than 0, tainted
- Or download: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/debugging/kernel-chktaint

### 46. What happens when you specify invalid license (Say "abc")
- When invalid license is used, module vericiation fails and kernel is tainted

### 47. What happens when a non-GPL kernel module trying to access GPL module
- If non-GPL license tries to use EXPORT_SYMBOL_GPL, compilation will fail

## Section 6: Module Metadata

### 48. How to find out kernel version from a .ko
- Finding version number: modinfo ./module1.ko
  - Find vermagic

### 49. Module metadata
- Module metadata
```
MODULE_DESCRIPTION("Hello World");
MODULE_AUTHOR("hpjeon");
MODULE_LICENSE("GPL");
MODULE_VERSION("1.1.1");
```
- Check through modinfo command
  - vermagic: strings are checked for matching
  - intree: When accecpted, in-tree. Initially out-of-tree
  - srcversion: MD4 hash of the source code
    - Can be used to check any patching or update
  - retpoline: against Spectre bug

### 50. MODULE_INFO macro
- MODULE_INFO macro
  - MODULE_VERSION(), MODULE_AUTHOR(), MODULE_LICENSE() actually call MODULE_INF()

### 51. Objdump on Kernel module
- objdump --section-headers ./hello.ko
- objdump --section-headers --section=.modinfo --full-contents ./hello.ko
```bash
./hello.ko:     file format elf64-x86-64

Sections:
Idx Name          Size      VMA               LMA               File off  Algn
  4 .modinfo      000000b4  0000000000000000  0000000000000000  000000f0  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
Contents of section .modinfo:
 0000 76657273 696f6e3d 312e312e 31006c69  version=1.1.1.li
 0010 63656e73 653d4750 4c006175 74686f72  cense=GPL.author
 0020 3d68706a 656f6e00 64657363 72697074  =hpjeon.descript
 0030 696f6e3d 48656c6c 6f20576f 726c6400  ion=Hello World.
 0040 73726376 65727369 6f6e3d38 46423246  srcversion=8FB2F
```

## Section 7: Printk

### 53. Printk kernel ring buffer size
- cat /boot/config-`uname -r` |grep CONFIG_LOG_BUF_SHIFT
  - 18 implies 256k

### 54. Printk Log Levels
- printk log level
- KERN_SOH is defined as \001
- dmesg -x shows the status of log level
```bash
printk("\001""4""%s: In init\n", __func__);
...
printk("\001""2""%s: In exit\n", __func__);
```
  - In the example of hello.c, init uses 4, which is warning. exit uses 2, which is critical
```bash
kern  :warn  : [140776.891848] test_hello_init: In init
kern  :crit  : [140812.192295] test_hello_exit: In exit
```

### 55. Default printk log level
- Without explicit log level, warn is the default for the printk
```bash
$ cat /proc/sys/kernel/printk
4	4	1	7
```
  - console log level/default message level/minimum console log level/maximum console loglevel

### 57. Short printk macros
- printk macro at /lib/modules/`uname -r`/build/include/linux/printk.h
  - pr_emerg(), pr_alert(), pr_crit(), pr_err(), pr_warn(), ...
  - They will configure log level as info, warn as used

### 58. Enable pr_debug messages
- pr_debug() works only when CONFIG_DYNAMIC_DEBUG or DEBUG exists
- Add `ccflags-y := -DDEBUG` in the Makefile

### 59. Linux Kernel Module example which prints floating point number

### 60.
- floating number is not supported in kernel space
- FPU is not necessary and might be expensive. So disabled in kernel
  - Some architecture may not have FPU

### 61. Limiting printk messages - printk_rate_limit
- Limiting printk messages
- cat /proc/sys/kernel/printk_ratelimit
  - Time limit in seconds: Ex) 5 (sec)
- cat /proc/sys/kernel/printk_ratelimit_burst
  - Number of limits: Ex) 10

### 62. Limiting printk messages - printk_once
- printk_once() will print once in the loop
  - KERN_INFO only
- If printk_once() is located other lines of the code, it will print

### 63. Avoiding default new line behavior of printk
- printk() will inject "\n" anyway

### 64. Printing hex dump - print_hex_dump
- print_hex_dump()

### 65. Printing hex dump - print_hex_dump_bytges
- print_hex_dump_bytes()

### 66. Dynamic Debug
- Dynamic Debug
  - CONFIG_DYNAMIC_DEBUG
    - check /boot/config-`uname -r`
  - dynamic_pr_debug()
    - pr_debug() prints too many logging info.
    - mount |grep debugfs
      - debugfs is mounted

## Section 8: Sysem call for loading module

## Section 9: Kernel Panic, Oops, Bug

### 73. Kernel Panic Example
- Kernel panic : an error in the kernel code
  - calls panic() function to dump debug info.
  - /proc/sys/kernel/panic : After N seconds, will reboot
- panic_test.c
  - Will reboot the OS

### 74. What is oops
- OOPS : similar to segfault in user space
  - Processor status
  - Contents of CPU registers
  - Call trace
- oops_test.c
```bash
[Fri Nov  6 10:00:50 2020] IP: test_oops_init+0x1c/0x30 [oops_test]
[Fri Nov  6 10:00:50 2020] PGD 0 P4D 0
[Fri Nov  6 10:00:50 2020] Oops: 0002 [#1] SMP PTI
```

### 76. What is BUG and example
- BUG_ON(condition) : macro in Linux device drivers
  - Same as `if (condition) BUG()`
- BUG()
  - prints the contents of the registers
  - prints stack trace
  - kill process
- bug_test.c
  - dmesg shows:
```
 kern  :crit  : [169315.139710] kernel BUG at /home/hpjeon/hw/class/udemy_100Lectures_on_LinuxKernelProgram/section_9/bug_test.c:7!
```

### 77. Can we remove module after bug/oops
- oops/BUG() module will not be able to be removed

## Section 10: Process Management in Linux Kernel

### 80. How to find out how many CPUs are present from user space and kernel space
- Counting the number of CPUs
- cat /proc/cpuinfo
- num_online_cpus() in online_cpus.c

### 81. Process representation in Linux Kernel and process states
- Linux kernel internally refers processes as tasks
- List of processes is stored as the task list, which is a circular doubly linked list
- /lib/modules/4.15.0-122-generic/build/include/linux/sched.h
  - struct task_struct {}
- State of a process
  - R: Running or on a run-queue
  - S: Sleeping/blocked. Can be runnable/awaken by a signal
  - D: Similar to S but not waken-up by a signal
  - T: Stopped
  - I: Idling
  - ps -el shows the states of processes

### 82. Linux Kernel Module example demonstrating process name, process id, and process s

### 83. Linux Kernel Module example demonstrating current macro
- Read task_struct of the corresponding process when a module needs info in the kernel
- current.c
  - Will print insmod when inserted. rmmod for removal.
  - current macro from <asm/current.h>

### 86. Process Memory Map
- Process memory map
- ps -ef # find id like 3637
- cat /proc/3637/maps
  - Those info are stored in struct mm_struct

## Section 11: Kernel Threads

### 88. Introduction to Kernel Threads
- Kernel thread: Linux task running in kernel mode only
- ps -ef resuls with []
```
root     18658     2  0 15:21 ?        00:00:00 [kworker/2:3]
root     18659     2  0 15:21 ?        00:00:00 [kworker/0:1]
hpjeon   18675  3652 20 15:21 pts/0    00:03:53 /usr/lib/firefox/firefox -privat
hpjeon   18738 18675 33 15:21 pts/0    00:06:20 /usr/lib/firefox/firefox -conten
```
- ps 18658 and 18659 are kernel threads
- Compared to user thread, kernel threads don't have address space. mm variable in task_struct {} is NULL

### 89. Kernel Thread Example - kthread_create
- kthread_create()/kthread_stop() in kthread.c

### 93. What happens if we don't use kthread_should_stop() in thread function
- If kthread_should_stop() is not used, process is killed when the module is removed

### 94. What happens if we don't call kthread_stop() in module_exit
- If kthread_stop() is not used, oops happens when rmmod is tried. The module will not be removed

## Section 12: Module Support for Multiple Kernels

### 97. LINUX_VERSION_CODE Macro
- LINUX_VERSION_CODE at /lib/modules/4.15.0-122-generic/build/include/generated/uapi/linux/version.h
- /lib/modules/4.15.0-122-generic/build/Makefile
```
VERSION = 4
PATCHLEVEL = 15
SUBLEVEL = 18
```
- LINUX_VERSION_CODE = 65536 * VERSION + 256 * PATCHLEVEL + SUBLEVEL

### 98. KERNEL_VERSION Macro
- KERNEL_VERSION macro

## Section 13: Bonus Section

### 103. Significance of __init macro
- __init will make the kernel to remove init function after loading the module, saving memory

### 105. __exit macro
- __exit is a macro from linux/init.h
- built-in  module doesn't need exit function. When built-in, __exit will disregard the exit function
- Good practice to have

### 106. __initdata and __exitdata macro
- __initdata/__exitdata: when __init is used in the init function

### 107. How do you list builtin modules
- The list of built-in modules at /lib/modules/`uname -r`/modules.builtin

### 109. Blacklisting Kernel Modules
- Blacklisting modules : blacklist <modulename>
  - /etc/modprobe.d/*.conf with blacklist keyword
  - sudo update-initramfs -u # updating initial ram disk

### 112. systool  
- systool -vm e1000
