#include "core/cpuvol.h"
#include "native_source.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  if (argc != 3)
    return 3;
  if (!strcmp(argv[1], "cpu")) {
    r3d_cpuvol v;
    if (r3d_cpuvol_open(&v, argv[2], 8))
      return 3;
    unsigned char data[4096];
    bool complete =
        r3d_cpuvol_read_block_status(&v, 0, 0, 0, 0, 16, 16, 16, data);
    r3d_cpuvol_close(&v);
    if (!complete)
      return 4;
    for (unsigned i = 0; i < sizeof data; i++)
      if (data[i] != 100)
        return 5;
    return 0;
  }
  CURL *c = curl_easy_init();
  bool downloaded=false;
  int state = r3d_native_fetch(c, argv[1], argv[2], 0, 0, 0, 0, &downloaded);
  if (state) {
    int cached=r3d_native_fetch(c,argv[1],argv[2],0,0,0,0,&downloaded);
    if(cached!=state || downloaded) state=3;
  }
  curl_easy_cleanup(c);
  return state;
}
