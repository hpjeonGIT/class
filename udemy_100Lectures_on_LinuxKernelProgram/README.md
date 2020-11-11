# Class title
- Learn Linux Kernel Programming by Linux Trainer
- Interfacing kernel modules using insmod/rmmod using printk, init, exit
- Actual kernel core programming is not addressed

## Lecture 1.2
- Kernal module: loadable. Not requiring rebooting of the system
- *.ko:kernel object
- /lib/modules/`uname-r`/kernel
  - More than 5000 ko files

## Lecture 1.3
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

## Lecture 1.5
- lsmod : info of /sys/modules (in old days, /proc/modules)
- modinfo: info of the module. Shows the list of parameters or arguments of the module.

## Lecture 1.6
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
  ## Lecture 1.7

- No comma in printk()
- log_prirority: EMERG, ALERT, CRIT, ERR, WARNING, NOTICE, INFO, DEBUG

## Lecture 1.8
- Update of Makefile
```
obj-m := hello.o
all:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean
```
- Make sure that the tab is injected in the first column

## Lecture 1.10
- Returning -1 from test_hello_init() will not be loaded in the kernel MODULE_LICENSE

## Lecture 1.11
- Renaming as mymodule.ko
```
obj-m := mymodule.o
mymodule-objs := hello.o
all:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean
```

## Lecture 1.12
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

## Lecture 1.13
- A single Make file and multiple modules
```
obj-m := hello1.o
obj-m += hello2.o
all:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean
```

## Lecture 1.14
- dmesg command
  - dmesg -c or dmesg -C to clear the ring buffer
  - dmesg -l err,warn,crit -T # human readable, printing err,warn, critical messages
- /var/log/kern.log

## Lecture 1.15
- Running dmesg as background
  - dmesg -w &
  - Can observe module print when loaded or unloaded in the CLI

## Lecture 1.16
- If the module source doesn't have module_exit(), rmmod will not work to remove the module
  - Reboot to remove the module

## Lecture 1.17
- Without module_init(), having module_exit() only works for insmod and rmmod

## Lecture 2.21-22
- hello.c -> hello.o and hello.mod.c -> hello.mod.o
  - hello.o and hello.mod.o -> hello.ko
- Module.symvers : external symbols defined in the module
- module.order : for multiple modules, it will list the order

## Lecture 2.23
- lsmod
- insmod : can load any modules
- modinfo
- rmmod
- modprobe : can load modules at /lib/modules/$(uname -r) only. Check and load dependency automatically

## Lecture 2.24
- depmod : generates modules.dep and map files
- /lib/modules/`uname -r`/modules.dep

## Lecture 2.26
- gcc alias. Check alias.c and alias_v.c

## Lecture 2.27
- Using init_module() and cleanup_module() instead of module_init() and module_exit()
- Check module_init_exit.c

## Lecture 3.28
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

## Lecture 3.29
- Parameter data type will be checked.

## Lecture 3.30
- modprobe reads /etc/modprobe.conf for parameters (not existing for Ubuntu)

## Lecture 3.33
- Passing array as input
- sudo insmod ./parameter_array.ko  param_array=4,5,8

## Lecture 4.35
- symbol: name space for memory. Could be a variable or a function
- Every kernel image has a symbol table
  - sudo cat /boot/System.map-4.15.0-122-generic

## Lecture 4.36
- How to export a symbol
  - When you define a new function in the module
  - EXPORT_SYMBOL or EXPORT_SYMBOL_GPL

## Lecture 4.37
- /boot/System.map vs /proc/kallsyms
  - /proc/kallsyms: contains symbols of dynamically loaded modules + built-in modules
  - /boot/System.map: only built-in modules

## Lecture 4.38
- sudo insmod symbol_export.ko
- sudo cat /boot/System.map-`uname -r` doesn't find print_jiffies
- sudo cat /proc/kallsyms  finds print_jiffies

## Lecture 4.39
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

## Lecture 5.43
- When MODULE_LICENSE is missing, verification fails and loading is disabled

## Lecture 5.44
- Tainted kernel: not supported by community. Debugging functionality and API calls are limited.
  - Using proprietary kernel module may taint kernel

## Lecture 5.45
- How to check the kernel is tainted or not
  - Check dmseg
  - Check /proc/sys/kernel/tainted. When larger than 0, tainted
- Or download: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/tools/debugging/kernel-chktaint

## Lecture 5.46
- When invalid license is used, module vericiation fails and kernel is tainted

## Lecture 5.47
- If non-GPL license tries to use EXPORT_SYMBOL_GPL, compilation will fail

## Lecture 6.48
- Finding version number: modinfo ./module1.ko
  - Find vermagic

## Lecture 6.49
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

## Lecture 6.50
- MODULE_INFO macro
  - MODULE_VERSION(), MODULE_AUTHOR(), MODULE_LICENSE() actually call MODULE_INF()

## Lecture 6.51
- objdump --section-headers ./hello.ko
- objdump --section-headers --section=.modinfo --full-contents ./hello.ko
```
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

## Lecture 7.53
- cat /boot/config-`uname -r` |grep CONFIG_LOG_BUF_SHIFT
  - 18 implies 256k

## Lecture 7.54
- printk log level
- KERN_SOH is defined as \001
- dmesg -x shows the status of log level
```
printk("\001""4""%s: In init\n", __func__);
...
printk("\001""2""%s: In exit\n", __func__);
```
  - In the example of hello.c, init uses 4, which is warning. exit uses 2, which is critical
```
kern  :warn  : [140776.891848] test_hello_init: In init
kern  :crit  : [140812.192295] test_hello_exit: In exit
```

## Lecture 7.55
- Without explicit log level, warn is the default for the printk
```
cat /proc/sys/kernel/printk
4	4	1	7
```
  - console log level/default message level/minimum console log level/maximum console loglevel

## Lecture 7.57
- printk macro at /lib/modules/`uname -r`/build/include/linux/printk.h
  - pr_emerg(), pr_alert(), pr_crit(), pr_err(), pr_warn(), ...
  - They will configure log level as info, warn as used

## Lecture 7.58
- pr_debug() works only when CONFIG_DYNAMIC_DEBUG or DEBUG exists
- Add `ccflags-y := -DDEBUG` in the Makefile

## Lecture 7.59-60
- floating number is not supported in kernel space
- FPU is not necessary and might be expensive. So disabled in kernel
  - Some architecture may not have FPU

## Lecture 7.61
- Limiting printk messages
- cat /proc/sys/kernel/printk_ratelimit
  - Time limit in seconds: Ex) 5 (sec)
- cat /proc/sys/kernel/printk_ratelimit_burst
  - Number of limits: Ex) 10

## Lecture 7.62
- printk_once() will print once in the loop
  - KERN_INFO only
- If printk_once() is located other lines of the code, it will print

## Lecture 7.63
- printk() will inject "\n" anyway

## Lecture 7.64
- print_hex_dump()

## Lecture 7.65
- print_hex_dump_bytes()

## Lecture 7.66
- Dynamic Debug
  - CONFIG_DYNAMIC_DEBUG
    - check /boot/config-`uname -r`
  - dynamic_pr_debug()
    - pr_debug() prints too many logging info.
    - mount |grep debugfs
      - debugfs is mounted

## Lecture 9.73
- Kernel panic : an error in the kernel code
  - calls panic() function to dump debug info.
  - /proc/sys/kernel/panic : After N seconds, will reboot
- panic_test.c
  - Will reboot the OS

## Lecture 9.74
- OOPS : similar to segfault in user space
  - Processor status
  - Contents of CPU registers
  - Call trace
- oops_test.c
```
[Fri Nov  6 10:00:50 2020] IP: test_oops_init+0x1c/0x30 [oops_test]
[Fri Nov  6 10:00:50 2020] PGD 0 P4D 0
[Fri Nov  6 10:00:50 2020] Oops: 0002 [#1] SMP PTI
```

## Lecture 9.76
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

## Lecture 9.77
- oops/BUG() module will not be able to be removed

## Lecture 10.80
- Counting the number of CPUs
- cat /proc/cpuinfo
- num_online_cpus() in online_cpus.c

## Lecture 10.81
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

## Lecture 10.82-83
- Read task_struct of the corresponding process when a module needs info in the kernel
- current.c
  - Will print insmod when inserted. rmmod for removal.
  - current macro from <asm/current.h>

## Lecture 10.86
- Process memory map
- ps -ef # find id like 3637
- cat /proc/3637/maps
  - Those info are stored in struct mm_struct

## Lecture 11.88
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

## Lecture 11.89
- kthread_create()/kthread_stop() in kthread.c

## Lecture 11.93
- If kthread_should_stop() is not used, process is killed when the module is removed

## Lecture 11.94
- If kthread_stop() is not used, oops happens when rmmod is tried. The module will not be removed

## Lecture 12.97
- LINUX_VERSION_CODE at /lib/modules/4.15.0-122-generic/build/include/generated/uapi/linux/version.h
- /lib/modules/4.15.0-122-generic/build/Makefile
```
VERSION = 4
PATCHLEVEL = 15
SUBLEVEL = 18
```
- LINUX_VERSION_CODE = 65536 * VERSION + 256 * PATCHLEVEL + SUBLEVEL

## Lecture 12.98
- KERNEL_VERSION macro

## Lecture 13.103
- __init will make the kernel to remove init function after loading the module, saving memory
