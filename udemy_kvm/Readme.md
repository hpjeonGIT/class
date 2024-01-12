## Title: Linux KVM for System Engineers
- Instructor: Yogesh Mehta

1. A Welcome and Course Introduction

2. What is a Hypervisor?
  - A hypervisor is computer SW, firmware or HW that creates and runs virtual machines. A computer on which a hypervisor runs one or more virtual machines is called a host machine, and each VM is called a guest machine. The hypervisor presents the guest OS with a virtual operating platform and maanages the execution of the guest OS.
  - Type 1: XEN, Hyper-V, VMWare ESXi, native/bare-metal, run directly on the hosts's HW to control the hardware and to mmanage guest OS
  - Type 2: VMware workstation, VirtualBox, abstracts guest OS from the host OS.
  
3. KMV Introductiona dn its Pros/Cons
  - Benefits/features
    - Performance and scalability
    - Lower cost
    - Secure
    - Supports live/offline VM migration
    - Full virtualization
    - Open source
    - Various filesystems support
    - Resource over-committing
  - Drawbacks
    - Supported by certain processor types: https://www.linux-kvm.org/page/Processor_support
    - Complex Networking
  - Type 1 or 2

4. LAB details and installation Pre-Reqs
  - Processor's virtualization extension capability: `grep -E 'svm|vmx' /proc/cpuinfo`
    - Intel VT can be disabled/enabled in the BIOS configuration
    - AMD-V cannot be disabled in the BIOS
  - Kernel is greater than 2.6.20: `uname -a`
          
5. KVM Installation
  - yum install virt-install qemu-kvm libvirt libvirt-python libguestfs-tools virt-install virt-manageer
  - systemctl enable libvirtd
  - systemctl start libvirtd
  - systemctl status libvirtd
  - modinfo kvm; modinfo kvm_intel
6. Configuring Networking for KVM
  - With the installation of libvirtd and its services create a virtual bridge interface virbr0 with network 192.168.122.0/24. In your setup there might be requirements to use a different network. We will tune the virbr0 and eth1
  - /etc/sysconfig/network-scripts/ifcfg-eth1
  ```
  TYPE=Ethernet
  BOOTPROTO=none
  NAME=eth1
  ONBOOT=yes
  BRIDGE=virbr0
  HWADDR=<MAC_ADDRESS>
  ```
  - /etc/sysconfig/network-scripts/ifcfg-virbr0
  ```
  TYPE=BRIDGE
  DEVICE=virbr0
  BOOTPROTO=none
  ONBOOT=yes
  IPADDR=192.168.1.10
  NETWMASK=255.255.255.0
  GATEWAY=192.168.1.1
  ````
  - Enable the IPv4 forwarding
    - echo net.ipv4.ip_forward=1 > /ur/lib/sysctl.d/60-libvirtd.conf
    - /sbin/sysctl -p /usr/lib/sysctl.d/60-libvirtd.conf
  - Configure firewall
    - firewall-cmd --permanent --diret --passthrough ipv4 -I FORWARD -i bridge0 -j ACCEPT
    - firewall-cmd --permanent --diret --passthrough ipv4 -I FORWARD -o bridge0 -j ACCEPT
    - firewall-cmd --relload
  - virsh net-list
  - virsh net-dumpxml default # make surebridgename, ip address
  - virsh net-edit default # then we can edit using vim. To apply the new change, reboot
       
7. Configuring Storage of KVM
  - By Default, libvirt-based commands use the directory /var/lib/libvirt/images on a virtualization host as an initial file system directory storage pool.
  ```
  vgcreate lab_kvm_storage /dev/sdb
  lvcreate -l +100%FREE -n lab_kvm_lv lab_kvm_storage
  mkf.xfs /dev/mapper/lab_kvm_sotrage -lab_kvm_lv
  ```
  - Then add the fstab entry
  ```
  /dev/mapper/lab_kvm_storage-lab_kvm_lv  /var/lib/libvirt/images xfs defaults 0 0
  ```
  - Or watch the demo

8. Guest Virtual Machine Creation using CLI & GUI
  - Check if guest OS is supported by KVM: `osinfo-query os`
  - Create a guest VM
  ``` 
  virt-install --network bridge:virbr0 --name testvm1 --os-variant=centos7.0 --ram=1024 --vcpus=1 \
  --disk path=/var/lib/libvirt/images/testvm1-s.qcow2,format=qcow2,bus=virtio,size=5 \
  --graphics non --location=/osmedia/CentOS-7-x86_64-DVD-1511.iso \
  --extra-args="console=tty0 console=ttyS0,115200" --check all=off
  ```
  - See Lecture for each argument/options
  - Watch demo for sequential steps
  - virt-manager is a GUI tool for VM management

9. Cloning a guest VM
  - Shutdown the running source VM: `virsh shutdown soruce_vm`
  - Clone the VM: `virt-clone --original source_vm --name target_vm -f /var/lib/libvirt/images/target_vm.qcow2`
  - Run the target VM: `virsh start <target_vm>`

10. Snapshot Creation and Restore of a guest VM
  - List the current snapshots: `virsh sanpshot-list ttestvm1`
  - Create a snpashot: `virsh snapsho-create-as --domain guest_vm --name "snapshot_name" --description "somedescription"; virsh snapshot-list testvm1`
  - To check the details of a snapshot: `virsh snapshot-list guest_vm; virsh snapshot-info --domain guest_vm --current`
  - To revert to a snapshot: `virsh shutdown --domain guest_vm; virsh snapshot-revert --domain guest_vm --snapshotname "snapshot_name" --running`
  - To delete a snapshot: `virsh snapshot-delete --domain guest_vm --snapshotname "snapshot_name"`
    
11. Backup a guest VM
  - Shutdown the running guest VM: `virsh shutdown guest_vm`
  - Backup of the disk-image: `cp -p /var/lib/libvirt/images/gesut-vm-image.qcow2 /path/to/backup/`
  - Backup of the configuration file: `cp -p /etc/libvirt/qemu/guest-vm.xml /path/to/backup/`
    
12. Expanding the Qemu disk size
  - Check the storage pools: `virsh pool-list` # find pool name
  - Check vol-list: `virsh vol-list lab_kvmstoragepool` # use the found pool name to find volume
  - Check the details: `virsh vol-info /var/lib/libvirt/images/testvm1-os.qcow2`
  - Expand the disk: `qemu-img resize /var/lib/libvirt/images/testvm1-os.qcow2 +1G`
  - Check the volume properties to make sure: `virsh vol-info /var/lib/libvirt/images/testvm1-os.qcow2`

13. Repair a corrupted Qemu disk
  - Shutdown the running guest VM: `virsh shutdown guest_vm`
  - Install libuestfs-tools: `yum install libguestfs-tools`
  - Using guestfish, perform QEMU image filesystem repair:
  ```
  guestfish -a /kvmstore/kvmbox1.img
  ><fs> run
  ><fs> list-filesystems
  ><fs> fsck xfs /dev/sda1
  ><fs> q
  ```
  - Start the KVM guest: `virsh start guest_vm`

14. Manage the guest VM
       
| command               | Remarks|
|-----------------------|--------|
| virsh list      <br>  virsh list --all  | List all running VMs  <br> List all running VMs regardless of state  |
| virsh start guest_vm  <br>  virsh shutdown guest_vm  <br> virsh reboot guest_vm | start/shutdown/reboot the vm |
| virsh suspend guest_vm <br>  virsh resume guest_vm |   suspend/resume VM        |
| virsh shutdown guest_vm <br> virsh undefine guest_vm <br> virsh destory guest_vm | Destory VM instance |
| virsh console guest_vm  <br> ctrl+] | Enter guest's console <br> Exit guests console|
| virsh autostart guest vm <br> virsh autostart --disable guest_vm | Enables autostart of the guest VM <br> Disables autostart of the guest VM|
| virsh dmuuid guest_vm | Gets the UUID of the domain/vm|

15. Increasing resources on guest VM
  - How to change vCPU/Mem
  - Shutdown the guest VM: `virsh shutdown guest_vm`
  - Update VM: `virsh edit guest_vm`
  - Boot the guest VM: `virsh start guest_vm`

16. Performance Monitoring and Troubleshooting
  - /var/log/libvirt/qemu/*.log
