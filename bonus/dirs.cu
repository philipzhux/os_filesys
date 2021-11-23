__device__ void init_root(FileSystem *fs)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  curr_dir_fd = 0;
  // insert root dir fcb
  u32 bf = find_best_fit(fs,1024);
  set_bitmap(fs, bf, 1024,1);
  SET_BLOCK_OFFSET(fcb_base,fd,bf);
  my_strcpy((char*)NAME(fcb_base,0),"/");
  SET_SIZE(fcb_base,i,1024);
  SET_CTIME(fcb_base,i,++gtime);
  SET_DSIZE(fcb_base,fd，0);
  IS_DIR(fcb_base,fd) = 1;
  u32 new_dir_block_offset = GET_BLOCK_OFFSET(fcb_base,new_fd);
  u16* new_fds_ptr = (u16*)(volume+(fs->FILE_BASE_ADDRESS+new_dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  *new_fds_ptr++ = curr_dir_fd;
  *new_fds_ptr = 1024;
}

__device__ void del_from_dir(FileSystem *fs, u16 dir_fd, u16 file_fd)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,dir_fd);
  u16* fds_ptr = (u16*)(volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  while((*fds_ptr)!=file_fd) fds_ptr++;
  u16* tar_ptr = fds_ptr;
  while((*fds_ptr)!=1024) fds_ptr++;
  /** copy last to target **/
  *tar_ptr = *(--fds_ptr);
  /** delete last **/
  *(fds_ptr) = 1024;
  SET_DSIZE(fcb_base,dir_fd,GET_DSIZE(fcb_base,dir_fd)-my_strlen((char*)NAME(fcb_base,file_fd))-1);
}

__device__ u32 mk_dir(FileSystem *fs,char *s);
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 new_fd = fs_insert_fcb(FileSystem *fs,char *s);
  SET_SIZE(fcb_base,fd，1024);
  IS_DIR(fcb_base,fd) = 1;
  u32 new_dir_block_offset = GET_BLOCK_OFFSET(fcb_base,new_fd);
  u16* new_fds_ptr = (u16*)(volume+(fs->FILE_BASE_ADDRESS+new_dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  (*new_fds_ptr++) = curr_dir_fd; //add curr_dir as the parent of the new dir
  *new_fds_ptr = 1024; //add end
  add_to_dir(FileSystem *fs, curr_dir_fd, new_fd); //add new dir to curr_dir
  return new_fd;
}

__device__ u32 get_parent_fd(FileSystem *fs, u32 base_dir_fd)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,base_dir_fd);
  u16* fds_ptr = (u16*)(volume+(fs->FILE_BASE_ADDRESS+new_dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  u32 parent = *fds_ptr;
  return parent;
}


__device__ void get_pwd(FileSystem *fs, char* buffer)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  char names[5][20];
  u32 dir_fd = curr_dir_fd;
  int count = 0;
  while(dir_fd)
  {
    my_strcpy(names[count++],(char*)NAME(fcb_base,dir_fd));
    dir_fd = get_parent_fd(fs,dir_fd);
  }
  my_strcpy(buffer,"/");
  for(int t=count-1;i>=0;t--){
    my_strcpy(buffer,names[t]);
  }
}


__device__ void add_to_dir(FileSystem *fs, u16 dir_fd, u16 file_fd)
{
  uchar* fcb_base = fs->volume + fs->SUPERBLOCK_SIZE;
  u32 dir_block_offset = GET_BLOCK_OFFSET(fcb_base,dir_fd);
  u16* fds_ptr = (u16*)(volume+(fs->FILE_BASE_ADDRESS+dir_block_offset*fs->STORAGE_BLOCK_SIZE));
  while((*fds_ptr)!=1024) fds_ptr++;
  *(fds_ptr++)=file_fd;
  *(fds_ptr)=1024;
  SET_DSIZE(fcb_base,dir_fd,GET_DSIZE(fcb_base,dir_fd)+my_strlen((char*)NAME(fcb_base,file_fd))+1);
}