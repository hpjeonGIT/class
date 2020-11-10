#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/delay.h>
MODULE_LICENSE("GPL");
static int hello_init(void)
{
  int i;
  for(i=0;i<20;i++) {
    if (printk_ratelimit()) {
      pr_info("Printing %d\n",i+1);
    }
    else {
      pr_info("Sleeping for 5 seconds\n");
    msleep(5000);
    }
  }
  return 0;
}
static void hello_exit(void)
{
    pr_info("Done\n");

}
module_init(hello_init);
module_exit(hello_exit);
