//#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/printk.h>
#include <linux/cpumask.h>
MODULE_LICENSE("GPL");
static int __init mod_init(void)
{
  pr_info("number of online cpu is %d\n", num_online_cpus());
  return 0;
}
static void __exit mod_exit(void)
{
}

module_init(mod_init);
module_exit(mod_exit);
