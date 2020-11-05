# Class title
- Learn Linux Kernel Programming by Linux Trainer


Lecture 2
- Kernal module: loadable. Not requiring rebooting of the system
- *.ko:kernel object
- /lib/modules/`uname-r`/kernel
  - More than 5000 ko files

Lecture 3
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

Lecture 5
- lsmod : info of /sys/modules (in old days, /proc/modules)
- modinfo: info of the module. Shows the list of parameters or arguments of the module.

Lecture 6
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

