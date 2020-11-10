#include <linux/kernel.h>
#include <linux/module.h>
MODULE_LICENSE("GPL");
static int myinit(void)
{
    pr_info("dump_stack myinit\n");
    dump_stack();
    pr_info("dump_stack after\n");
    return 0;
}
static void myexit(void)
{
    pr_info("myexit\n");
}

module_init(myinit);
module_exit(myexit);
