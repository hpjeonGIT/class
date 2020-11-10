//#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <asm/current.h>
MODULE_LICENSE("GPL");
static int current_init(void)
{
  pr_info("current pid:%d, current process: %s\n",
          current->pid, current->comm);
  return 0;
}
static void current_exit(void)
{
  pr_info("current pid:%d, current process: %s\n",
          current->pid, current->comm);
}

module_init(current_init);
module_exit(current_exit);
