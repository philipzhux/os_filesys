#include "file_system.h"

__device__ void init_root(FileSystem *fs)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  curr_dir_fd = 0;
  // insert root dir fcb
  u32 bf = find_best_fit(fs,MAX_PER_DIR/sizeof(u16));
  set_bitmap(fs, bf, MAX_PER_DIR/sizeof(u16),1);
  SET_BLOCK_OFFSET(fcb_base,0,bf);
  my_strcpy((char*)NAME(fcb_base,0),"/");
  SET_SIZE(fcb_base,0,MAX_PER_DIR/sizeof(u16));
  SET_CTIME(fcb_base,0,++gtime);
  SET_DSIZE(fcb_base,0,0);
  IS_DIR(fcb_base,0) = 1;
  u32 root_block_offset = GET_BLOCK_OFFSET(fcb_base,0);
  u16* root_ptr = (u16*)(fs->volume+(fs->FILE_BASE_ADDRESS+root_block_offset*fs->STORAGE_BLOCK_SIZE));
  *root_ptr = curr_dir_fd;
  root_ptr++;
  *root_ptr = DIR_END;
}

__device__ void del_from_dir(FileSystem *fs, u16 dir_fd, u16 file_fd)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  SET_MTIME(fcb_base,dir_fd,++gtime);
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,dir_fd);
  u16* fds_ptr = (u16*)(fs->volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  while((*fds_ptr)!=file_fd) fds_ptr++;
  u16* tar_ptr = fds_ptr;
  while((*fds_ptr)!=DIR_END) fds_ptr++;
  /** copy last to target **/
  *tar_ptr = *(--fds_ptr);
  /** delete last **/
  *(fds_ptr) = DIR_END;
  SET_DSIZE(fcb_base,dir_fd,GET_DSIZE(fcb_base,dir_fd)-my_strlen((char*)NAME(fcb_base,file_fd))-1);
}

__device__ u32 mk_dir(FileSystem *fs,char *s)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 new_fd = fs_insert_fcb(fs,s);
  u32 bf = find_best_fit(fs,MAX_PER_DIR/sizeof(u16));
  set_bitmap(fs, bf, MAX_PER_DIR/sizeof(u16),1);
  SET_BLOCK_OFFSET(fcb_base,new_fd,bf);
  SET_SIZE(fcb_base,new_fd,MAX_PER_DIR/sizeof(u16));
  IS_DIR(fcb_base,new_fd) = 1;
  u32 new_dir_block_offset = GET_BLOCK_OFFSET(fcb_base,new_fd);
  u16* new_fds_ptr = (u16*)(fs->volume+(fs->FILE_BASE_ADDRESS+new_dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  *new_fds_ptr = curr_dir_fd; //add curr_dir as the parent of the new dir
  new_fds_ptr++;
  *new_fds_ptr = DIR_END; //add end
  add_to_dir(fs, curr_dir_fd, new_fd); //add new dir to curr_dir
  return new_fd;
}

__device__ u32 get_parent_fd(FileSystem *fs, u32 base_dir_fd)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,base_dir_fd);
  u16* fds_ptr = (u16*)(fs->volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  u32 parent = *fds_ptr;
  //printf("Parent of %d is %d\n",base_dir_fd,parent);
  return parent;
}


__device__ void get_pwd(FileSystem *fs, char* buffer)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  char** names = new char*[5];
  u32 dir_fd = curr_dir_fd;
  int count = 0;
  while(dir_fd)
  {
    //printf("Test: current layer %s\n",(char*)NAME(fcb_base,dir_fd));
    names[count++] = (char*)NAME(fcb_base,dir_fd);
    dir_fd = get_parent_fd(fs,dir_fd);
  }
  *buffer++ = '/';
  for(int t=count-1;t>=0;t--){
    char* ptr = names[t];
    while(*ptr) *buffer++ = *ptr++;
    if(t) *buffer++='/';
  }
  *buffer = '\0';
  free(names);
}


__device__ void add_to_dir(FileSystem *fs, u16 dir_fd, u16 file_fd)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  SET_MTIME(fcb_base,dir_fd,++gtime);
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,dir_fd);
  u16* fds_ptr = (u16*)(fs->volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  while((*fds_ptr)!=DIR_END) fds_ptr++;
  *(fds_ptr++)=file_fd;
  *(fds_ptr)=DIR_END;
  SET_DSIZE(fcb_base,dir_fd,GET_DSIZE(fcb_base,dir_fd)+my_strlen((char*)NAME(fcb_base,file_fd))+1);
}


__device__ u16 dir_count(FileSystem *fs, u16 dir_fd)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,dir_fd);
  u16* fds_ptr = (u16*)(fs->volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  u16 count = 0;
  while((*fds_ptr)!=DIR_END)
  {
    fds_ptr++;
    count++;
  }
  return count;
}