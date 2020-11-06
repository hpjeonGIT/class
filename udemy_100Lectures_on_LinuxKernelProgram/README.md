# Class title
- Learn Linux Kernel Programming by Linux Trainer


## Lecture 2
- Kernal module: loadable. Not requiring rebooting of the system
- *.ko:kernel object
- /lib/modules/`uname-r`/kernel
  - More than 5000 ko files

## Lecture 3
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

## Lecture 5
- lsmod : info of /sys/modules (in old days, /proc/modules)
- modinfo: info of the module. Shows the list of parameters or arguments of the module.

## Lecture 6
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

## Lecture 7
- No comma in printk()
- log_prirority: EMERG, ALERT, CRIT, ERR, WARNING, NOTICE, INFO, DEBUG

## Lecture 8
- Update of Makefile
```
obj-m := hello.o
all:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean
```
- Make sure that the tab is injected in the first column

## Lecture 10
- Returning -1 from test_hello_init() will not be loaded in the kernel MODULE_LICENSE

## Lecture 11
- Renaming as mymodule.ko
```
obj-m := mymodule.o
mymodule-objs := hello.o
all:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean
```

## Lecture 12
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

## Lecture 13
- A single Make file and multiple modules
```
obj-m := hello1.o
obj-m += hello2.o
all:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean
```

## Lecture 14
- dmesg command
  - dmesg -c or dmesg -C to clear the ring buffer
  - dmesg -l err,warn,crit -T # human readable, printing err,warn, critical messages
- /var/log/kern.log

## Lecture 15
- Running dmesg as background
  - dmesg -w
  - Can observe module print when loaded or unloaded in the CLI

## Lecture 16
- If the module source doesn't have module_exit(), rmmod will not work to remove the module
  - Reboot to remove the module

## Lecture 17
- Without module_init(), having module_exit() only works for insmod and rmmod

## Lecture 21-22
- hello.c -> hello.o and hello.mod.c -> hello.mod.o
  - hello.o and hello.mod.o -> hello.ko
- Module.symvers : external symbols defined in the module
- module.order : for multiple modules, it will list the order

## Lecture 23
- lsmod
- insmod : can load any modules
- modinfo
- rmmod
- modprobe : can load modules at /lib/modules/$(uname -r) only. Check and load dependency automatically

## Lecture 24
- depmod : generates modules.dep and map files
- /lib/modules/`uname -r`/modules.dep

## Lecture 26
- gcc alias. Check alias.c and alias_v.c

## Lecture 27
- Using init_module() and cleanup_module() instead of module_init() and module_exit()
- Check module_init_exit.c
