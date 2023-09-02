mkdir /ramdisk

# rd_size in kilobytes
modprobe brd rd_nr=1 rd_size=$((1024 * 1024))
mkfs.ext4 /dev/ram0
mount /dev/ram0 /ramdisk

# Unmount:
# umount /ramdisk
# modprobe -r brd
