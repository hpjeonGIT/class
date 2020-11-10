#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/delay.h>
MODULE_LICENSE("GPL");
static int hello_init(void)
{
  int i;
  for(i=0;i<20;i++) {
    printk_once(KERN_INFO"Printing %d\n",i);
  }
  return 0;
}
static void hello_exit(void)
{
    pr_info("Done\n");

}
module_init(hello_init);
module_exit(hello_exit);
