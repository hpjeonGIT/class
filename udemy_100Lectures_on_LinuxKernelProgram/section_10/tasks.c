#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sched/signal.h>
char buffer[256];
char *get_task_state(long state)
{
  switch (state) {
  case TASK_RUNNING:
    return "TASK RUNNING";
  case TASK_INTERRUPTIBLE:
    return "TASK_INTERRUPTIBLE";
  case TASK_UNINTERRUPTIBLE:
    return "TASK_UNINTERRUPTIBLE";
  case __TASK_STOPPED:
    return "__TASK_STOPPED";
  case __TASK_TRACED:
    return "__TASK_TRACED";
  default:
  {
    sprintf(buffer, "Unknown Type:%ld\n", state);
    return buffer;
  } }
}
MODULE_LICENSE("GPL");
static int test_tasks_init(void)
{
  struct task_struct *task_list;
  unsigned int process_count =0;
  pr_info("%s: In init\n", __func__);
  for_each_process(task_list) {
    pr_info("Process: %s\t PID:[%d]\t State:%s\n",
            task_list->comm, task_list->pid,
            get_task_state(task_list->state));
    process_count ++;
  }
  pr_info("Number of processes:%u\n", process_count);
  return 0;
}
static void test_tasks_exit(void)
{
  pr_info("%s: In exit\n", __func__);
}

module_init(test_tasks_init);
module_exit(test_tasks_exit);
