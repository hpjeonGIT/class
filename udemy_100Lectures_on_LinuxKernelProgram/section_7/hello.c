#include <linux/kernel.h>
#include <linux/module.h>
MODULE_LICENSE("GPL");
static int test_hello_init(void)
{
    printk(KERN_DEBUG"%s: In _init 1\n", __func__);
    pr_debug("%s: In init 2\n", __func__);
    return 0;
}
static void test_hello_exit(void)
{
    pr_warn("%s: In exit 1\n", __func__);
    pr_debug("%s: In exit 2\n", __func__);
}

module_init(test_hello_init);
module_exit(test_hello_exit);
