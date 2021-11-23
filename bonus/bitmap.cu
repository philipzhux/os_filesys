#include "file_system.h"

__device__ unsigned char block_available(FileSystem *fs, u32 offset)
{
  if(offset>(1<<15)) return 0;
  u32 map_byte = offset/8;
  u32 map_bit = offset%8;
  return (((fs->volume[map_byte]>>map_bit) & 1)==0);
}

__device__ void set_bitmap(FileSystem *fs, u32 offset, u32 size,unsigned char t) 
{
  u32 block_size = (size/fs->STORAGE_BLOCK_SIZE)+(size%fs->STORAGE_BLOCK_SIZE>0);
  for(u32 i=offset;i<offset+block_size;i++) set_bit(fs,i,t);
}

__device__ void set_bit(FileSystem *fs, u32 offset, unsigned char t) {
  u32 map_byte = offset/8;
  u32 map_bit = offset%8;
  if(t){
    fs->volume[map_byte] = fs->volume[map_byte] | (0x1<<map_bit);
    return;
  }
  fs->volume[map_byte] = fs->volume[map_byte] & ~(0x1<<map_bit);
  return;
}