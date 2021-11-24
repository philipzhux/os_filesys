# CSC 3150 Project 4 Report
**Zhu Chuyan** 119010486 **at** *The Chinese University of Hong Kong, Shenzhen*
## Table of contents
- [CSC 3150 Project 4 Report](#csc-3150-project-4-report)
  - [Table of contents](#table-of-contents)
  - [1. How to run the program](#1-how-to-run-the-program)
    - [1.1 Running environments](#11-running-environments)
    - [1.2 Executions](#12-executions)
      - [Main task: Single layer filesystem](#main-task-single-layer-filesystem)
      - [Bonus: Multiple Process VM Simulation](#bonus-multiple-process-vm-simulation)
  - [2. Implementations](#2-implementations)
    - [2.1 Single Layer Filesystem Simulation](#21-single-layer-filesystem-simulation)
      - [Implementation Flowchart](#implementation-flowchart)
      - [Bitmap](#bitmap)
      - [File Control Block](#file-control-block)
      - [Storage Block](#storage-block)
      - [Allocation: Find Best Fit](#allocation-find-best-fit)
      - [Compaction: External Framentation Elimation](#compaction-external-framentation-elimation)
      - [Basic operations support](#basic-operations-support)
      - [Memory Layouts](#memory-layouts)
    - [2.2 Filesys with multi-layer directories (Bonus)](#22-filesys-with-multi-layer-directories-bonus)
      - [Directory Implementation](#directory-implementation)
      - [Basic operations support](#basic-operations-support-1)
  - [3. The results](#3-the-results)
    - [Basic Part](#basic-part)
      - [Test 1](#test-1)
      - [Test 2](#test-2)
      - [Test 3](#test-3)
    - [Bonus Part](#bonus-part)
  - [4. What I learn from the project](#4-what-i-learn-from-the-project)


## 1. How to run the program
### 1.1 Running environments
* Operating system: ```Linux```
* Linux distribution: ```CentOS Linux release 7.6.1810 (Core)```
* Linux kernel version: ```Linux version 3.10.0-957.21.3.el7.x86_64```
* Compiler/CUDA version: ```Cuda compilation tools, release 10.1, V10.1.105```

### 1.2 Executions
#### Main task: Single layer filesystem
A Makefile has been prepared, so you can compile with the following command:
```shell
cd path_to_project/source/
make
```
You can then test the program by executing
```shell
./fs
```
#### Bonus: Multiple Process VM Simulation
```shell
cd path_to_project/bonus/
make
# if you wish to clean the object files:
make clean
```
Then you can simply run by:
```shell
./fsplus
```

## 2. Implementations
### 2.1 Single Layer Filesystem Simulation

#### Implementation Flowchart
 <figure align="center">

  ```mermaid
  graph LR;
  BITMAP["bitmap"];
  subgraph File Control Block
  FD["index (fd/fp)"];
  OFFSET["block_offset"];
  MT["mtime"];
  CT["ctime"];
  SIZE["size"];
  NAME["name"];
  end
  subgraph Storage Blocks
  BLOCKS["Blocks"];
  end
  subgraph Operations
  NEWO["Open file (Create)"];
  OPEN["Open existing"];
  RW["Read/Write"];
  end
  BITMAP--"Availability"-->BLOCKS;
  OFFSET--"Locates"-->BLOCKS;
  NEWO--"Find vacant slots"-->BITMAP;
  NEWO--"Create"-->NAME;
  FD--"Return to"-->NEWO;
  OPEN--"Search for"-->NAME;
  FD--"Return to"-->OPEN;
  NEWO-->CT;
  RW--"Feed FD"-->FD;
  FD--"Get offset"-->RW;
  RW--"Update"-->MT;
  RW--"Read/Write using offset"-->BLOCKS;
  ```

  <figcaption>Figure 1: Main Program Implementation Flow Chart</figcaption>
  </figure>

***
#### Bitmap

A bitmap is maintained in the SB (super block). **Each bit in the area is mapped to each block in the storage area, with 0 being vacant and 1 being occupied.** Simple bitwise operations and arithmetic are used to construct and query the map, and example is as simple as:
```C++
inline unsigned char block_available(FS *fs, u32 offset)
{
  if(offset>(1<<15)) return 0;
  u32 map_byte = offset/8;
  u32 map_bit = offset%8;
  return (((fs->volume[map_byte]>>map_bit) & 1)==0);
}
```
***

#### File Control Block
The file control block stores the metadata related to a specific file/directory, with attributes including:
* *name*: the file/directory name with length less than 20 bytes
* *block offset*: the block offset of the storage location
* *size*: the size of the file/directory in bytes
* *ctime*: the create time of the file/directory
* *mtime*: the modified time of the file/directory

Note that the FCB of a file/directory is unique regardless of the name, therefore the **index**, or **pointer**, of the FCB is used as the identifier of the file, named as file pointer(fs) or file descriptor(fd), which is returned upon invocation of ```open```.
***
#### Storage Block
The storage block is aligned as block (sized 32 bytes), which is actually a chunk of contiguous memory in the ```volume``` array. It can be accessed through the block offset given by the FCB.
***
#### Allocation: Find Best Fit
A **contiguous** storage allocation is required to be implemented. Therefore, to reduce external framentation upon storage allocation, the best fit allocaiton policy is adpoted due to the limited number of files (therfore running time).

The find-best-fit implementation simply scan through the bitmap from the start to the end, implementing as follows:
* It will count the length vacant chunks it comes accross and record
* If the chunk is **greater or equal than the needed**, and **strictly less than** the previous best-fit, then take it as the new best-fit
* Running the loop until the end

With such implemetation, the best fit possible allocation is achieved.
***
#### Compaction: External Framentation Elimation
Despite the use of best-fit policy, the external fragmentation is impossible to fully avoid. Therefore, in case that the storage space is insufficient, a compation is conducted,whose implementation is as follows:
* Two pointer is maintained: ```vacant_front``` and ```chunk_front```
* Initially, the vacant_front will scanning from the top of the bit map until it reaches a vacant block;
* Then, ```chunk_front``` will scan from the position of ```vacant_front``` until it reaches a occupied block
* Move the chunk starting at ```chunk_front``` to ```vacant_front```, update the corresponding FCB (no direct mappings, therefore need to search through the FCB entries to find the corresponding FCB)
* Move the ```vacant_front``` to the next block pass the chunk, and continue the above process until ```chunk_front``` goes beyond the storage area
***
#### Basic operations support
* open: simply search the fcb by name and return the fcb index as the file descriptor, if not existing then create a new fcb with 0 size and return the fd.
* read/write: simply query the block offset in FCB using fd and use the block offset to locate the storage and read/write; in case of size change upon writing, allocate a new storage chunk and free the origin
* ls: simply go through all the FCBs, and extract those whose name is not empty (```name[0]!='\0'```)
* rm: simply rm the FCBs and free the corresponding blocks in the bitmap (set as vacant)
***
#### Memory Layouts
To fulfills the designing ideas mentioned above, the **memory layout** is shown as below and dozen of **macros** are written to facilitate the use.
![mem_new](https://i.imgur.com/oU3Hl7E.png)

Some componenents are elaborated as follows:
* ctime: the create time of the file/dir
* mtime: the modiefed time of the file/dir
* size: the actual size of the file
* dsize: the directory size according to the rule *(bonus)*
* is_dir: whether the file is a directory of regular file *(bonus)*
***


### 2.2 Filesys with multi-layer directories (Bonus)

A multiple layer of directories is actually quite easy to achieve based on the basic implementations above.

#### Directory Implementation
Similar to the concept in Linux and many other major OSes, directory is simply a file recording its members (files/subdir), its predecessor (enclosing dir, like ```..``` in linux), and itself (like ```.``` in linux)). Here there is no need to maintain the info of "itself".

Therefore, a directory in my implmentation, like any other file, has a FCB with is_dir set as 1. The content of the file is the **fd of enclosing dir** and **fd of its members, including subdirs**. The start of the content is always the **fd of enclosing dir**. 

**An illustration**:
![dirill](https://i.imgur.com/YI6Mufm.png)

Therefore, there is indeed no much difference than the basic task, except a global variable, ```curr_dir_fd``` is needed to record of fd of current directory, and the root directory, namely ```/``` is created upon initialization.

To meet with the requirement that the size of directory is the sum of the length of names of its members, an extra metadata attribute ```dsize``` is used to hack it, which is actually not the real size of the directory (keeping name of member is rather unreasonable from my point of view).
***
#### Basic operations support
* open: same as the basic task, but instead of searching all fcb entries (searching from fd 0 to upper bound), the current directory file is query and only the its member fds will be inspected and compared with name (searching fds including in the current dir's directory file).

* read/write: exactly the same as the basic part
* ls: simply go through ~~all the FCBs~~ all fds including in the current dir's directory file, and extract those whose name is not empty (```name[0]!='\0'```)
* rm_rf:
  * for regular files: simply rm the FCBs and free the corresponding blocks in the bitmap (set as vacant), and at the same time remove the corresponding fd from the current dir file
  * for directories: recersively call rm_rf for all its members and finally delete itself like a regular file

* pwd: simply use the curr_dir_fd to query the FCB to get the name of current dir
* cd: simply search for (same search strategy as above to only cover those in the current dir) the fd of destination dir, set the curr_dir_fd to it.
* cd_p: go to the directory file of the current dir, the top entry (see the  illustration figure above) is the fd of parent dir, then simply set the curr_dir_fd to it.

To be easier (lazier), I did not allow the size (real one, not the dsize) of the directory to be dynamically adjusted (which requires copy and move of the directory file when new files created or deleted), but make it fixed. In my configuration, 255 files are supported in a single directory (256 including the parent, more than the required 50). It can be changed to maximum 511 by setting the MAX_PER_DIR to 512 in ```config.h```. *( because 1024/sizeof(u16)=512 )*
Also, with such implmentation, the layer of directory is not limited.

## 3. The results
### Basic Part
#### Test 1
![ss1](https://i.imgur.com/Qnkam2y.png)
#### Test 2
![ss2](https://i.imgur.com/Pj7YBpl.png)
#### Test 3
![ss3](https://i.imgur.com/iQ8zygq.png)

### Bonus Part
![bonus](https://i.imgur.com/vVeZcpq.png)![bonus2](https://i.imgur.com/bkVWiPI.png)

## 4. What I learn from the project
I learn the principles and implementations of a contiguous storage allocation file system and learn how to implement the directories, arrange the storage, maintain the metadatas, as well as how to reduce the external fragmentation using allocating policies and compaction.