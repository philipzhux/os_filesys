#include "file_system.h"

__device__ void compact_disk(FileSystem *fs)
{
  u32 fd;
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 vacant_front,chunk_front = 0;
  while(chunk_front<REAL_SOTRAGE_BLOCKS) {
    while(!block_available(fs,vacant_front) && vacant_front<REAL_SOTRAGE_BLOCKS) vacant_front++;
    chunk_front = vacant_front;
    while(chunk_front<REAL_SOTRAGE_BLOCKS && block_available(fs,chunk_front)) chunk_front++;
    if(chunk_front<REAL_SOTRAGE_BLOCKS && (fd = offset_to_fd(fs,chunk_front))<1024){
      int size = GET_SIZE(fcb_base,fd);
      for(int i=0;i<size;i++) fs->volume[fs->FILE_BASE_ADDRESS+vacant_front*
      fs->STORAGE_BLOCK_SIZE+i] = fs->volume[fs->FILE_BASE_ADDRESS+chunk_front*
      fs->STORAGE_BLOCK_SIZE+i];
      set_bitmap(fs,chunk_front,size,0);
      set_bitmap(fs,vacant_front,size,1);
      SET_BLOCK_OFFSET(fcb_base,fd,vacant_front);
      vacant_front += (size/fs->STORAGE_BLOCK_SIZE)+(size%fs->STORAGE_BLOCK_SIZE>0);
    }
  }
}

__device__ u32 offset_to_fd(FileSystem *fs, u32 offset)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 fd;
  for(fd=0;fd<1024;fd++){
    if(GET_BLOCK_OFFSET(fcb_base,fd)==offset) break;
  }
  return fd;
}

__device__ u32 find_best_fit(FileSystem *fs, int size)
{
  int best_fit = REAL_SOTRAGE_BLOCKS;
  int bf_blocks = REAL_SOTRAGE_BLOCKS;
  int block_needed = (size/fs->STORAGE_BLOCK_SIZE)+(size%fs->STORAGE_BLOCK_SIZE>0);
  int block = 0;
  int chunk = 0;
  int cursor = 0;
  while(block<REAL_SOTRAGE_BLOCKS)
  {
    if(!block_available(fs,block))
    {
      while(!block_available(fs,block)) block++;
      chunk = 0;
      cursor = block;
      continue;
    }
    if(chunk>=block_needed && chunk<bf_blocks)
    {
      best_fit = cursor;
      bf_blocks = chunk;
    }
    block++;
    chunk++;
  }
  return best_fit;
}


__device__ void rm_rf(FileSystem *fs, u32 fd)
{
    uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
    if(!IS_DIR(fcb_base,fd)){
      fs_delete_fd(fs,fd);
      del_from_dir(fs,curr_dir_fd,fd);
      return;
    }
    u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,fd);
    u16* fds_ptr = (u16*)(fs->volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
    fds_ptr++; //avoid deleting the parent dir
    while((*fds_ptr)!=1024) {
      rm_rf(fs,(*fds_ptr));
      fds_ptr++;
    }
    /* finish deleting subdir and files, delete self */
    fs_delete_fd(fs,fd);
    del_from_dir(fs,curr_dir_fd,fd);
}


__device__ void fs_delete_fd(FileSystem *fs,u32 fd)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  set_bitmap(fs, GET_BLOCK_OFFSET(fcb_base,fd), GET_SIZE(fcb_base,fd),0);
  NAME(fcb_base,fd)[0] = '\0';
  SET_SIZE(fcb_base,fd,0);
}

__device__ u32 fs_insert_fcb(FileSystem *fs,char *s)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  for(u32 i=0; i<1024; i++) {
    if(NAME(fcb_base,i)[0]=='\0'){
      my_strcpy((char*)NAME(fcb_base,i),s);
      SET_SIZE(fcb_base,i,0);
      SET_CTIME(fcb_base,i,++gtime);
      SET_MTIME(fcb_base,i,gtime);
      SET_DSIZE(fcb_base,i,0);
      IS_DIR(fcb_base,i) = 0;
      return i;
    }
  }
  return 1024;
}
__device__ u32 fs_search(FileSystem *fs,char *s)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,curr_dir_fd);
  u16* fds_ptr = (u16*)(fs->volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  fds_ptr++; //avoid intervening of the parent node
  while((*fds_ptr)!=1024){
    if(my_strcmp((char*)NAME(fcb_base,*fds_ptr),s)==0) return (*fds_ptr);
    fds_ptr++;
  }
  return 1024;
}