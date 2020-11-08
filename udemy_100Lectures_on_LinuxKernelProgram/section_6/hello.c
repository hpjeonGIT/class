#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/delay.h>
static int test_hello_init(void)
{
    printk(KERN_INFO"%s: In init\n", __func__);
    return 0;
}
static void test_hello_exit(void)
{
    printk(KERN_INFO"%s: In exit\n", __func__);
}

module_init(test_hello_init);
module_exit(test_hello_exit);
MODULE_DESCRIPTION("Hello World");
MODULE_AUTHOR("hpjeon");
MODULE_LICENSE("GPL");
MODULE_VERSION("1.1.1");
