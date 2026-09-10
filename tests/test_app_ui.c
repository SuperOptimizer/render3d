/* Exercise the actual application cache and list implementation without a
 * GPU context. Including the app keeps private cache ownership private. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#define main r3d_application_main
#include "../src/main.c"
#undef main

static void wait_cache(sgcache *c, uint32_t index, bool activation) {
  uint64_t end = r3d_now_ns() + 3000000000ull;
  for (;;) {
    pthread_mutex_lock(&c->mu);
    bool ready =
        activation ? c->act_ready == index : c->ent[index].state == SGC_READY;
    pthread_mutex_unlock(&c->mu);
    if (ready)
      return;
    assert(r3d_now_ns() < end);
    struct timespec delay = {.tv_nsec = 1000000};
    nanosleep(&delay, NULL);
  }
}
static void corpus(void) {
  char tmp[] = "/tmp/r3d-app-ui-XXXXXX";
  assert(mkdtemp(tmp));
  char a[160], b[160], store[160];
  snprintf(a, sizeof a, "%s/a", tmp);
  snprintf(b, sizeof b, "%s/b", tmp);
  snprintf(store, sizeof store, "%s/store", tmp);
  float xyz[32 * 32 * 3];
  for (uint32_t y = 0; y < 32; y++)
    for (uint32_t x = 0; x < 32; x++) {
      size_t k = ((size_t)y * 32 + x) * 3;
      xyz[k] = (float)x + 48;
      xyz[k + 1] = (float)y + 48;
      xyz[k + 2] = 128;
    }
  assert(flat_save(a, xyz, 32, 32, 1) == 0 &&
         flat_save(b, xyz, 32, 32, 1) == 0);
  assert(mkdir(store, 0755) == 0);
  const char *dirs[] = {b, a};
  assert(r3d_segstore_build(store, dirs, 2, 2, false) == 2);
  sgcache cache;
  assert(sgc_open(&cache, store, 1) == 0);
  assert(strcmp(cache.st.segs[cache.order[0]].name, "a") == 0);
  pthread_mutex_lock(&cache.mu);
  sgc_request(&cache, 0);
  pthread_mutex_unlock(&cache.mu);
  wait_cache(&cache, 0, false);
  pthread_mutex_lock(&cache.mu);
  cache.ent[0].pins++;
  float *leased = cache.ent[0].s.xyz;
  sgc_request(&cache, 1);
  pthread_mutex_unlock(&cache.mu);
  wait_cache(&cache, 1, false);
  /* A worker completed another decode under a one-byte budget while the
   * GUI held a grid lease. The old grid is still readable and pinned. */
  pthread_mutex_lock(&cache.mu);
  assert(cache.ent[0].state == SGC_READY && cache.ent[0].s.xyz == leased);
  assert(leased[0] >= 0 && cache.ready == 2);
  pthread_mutex_unlock(&cache.mu);
  sgc_activate(&cache, 1);
  wait_cache(&cache, 1, true);
  pthread_mutex_lock(&cache.mu);
  cache.ent[0].pins--;
  sgc_evict_lru(&cache, UINT32_MAX);
  assert(cache.bytes <= cache.budget && cache.ready == 0);
  pthread_mutex_unlock(&cache.mu);
  sgc_close(&cache);
  if (getenv("R3D_APP_UI_KEEP")) {
    printf("app UI fixture: %s\n", tmp);
    return;
  }
  char command[200];
  snprintf(command, sizeof command, "rm -rf '%s'", tmp);
  assert(system(command) == 0);
}
static void clipped_list(void) {
  enum { N = 10000 };
  sgc_sort_entry *sort = malloc(N * sizeof *sort);
  char (*names)[16] = malloc(N * sizeof *names);
  assert(sort && names);
  for (uint32_t i = 0; i < N; i++) {
    snprintf(names[i], 16, "%08u", N - i);
    sort[i] = (sgc_sort_entry){names[i], i};
  }
  qsort(sort, N, sizeof *sort, sgc_sort_compare);
  for (uint32_t i = 1; i < N; i++)
    assert(strcmp(sort[i - 1].name, sort[i].name) < 0);
  assert(sort[0].index == N - 1 && sort[N - 1].index == 0);
  ImGuiContext *ctx = igCreateContext(NULL);
  ImGuiIO *io = igGetIO_Nil();
  io->DisplaySize = (ImVec2){640, 480};
  io->DeltaTime = 1.0f / 60.0f;
  io->IniFilename = NULL;
  io->BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
  ImGuiListClipper *clipper = ImGuiListClipper_ImGuiListClipper();
  for (int frame = 0; frame < 3; frame++) {
    igNewFrame();
    igSetNextWindowSize((ImVec2){400, 300}, ImGuiCond_Always);
    igBegin("corpus", NULL, 0);
    int rows = 0;
    ImGuiListClipper_Begin(clipper, N, -1);
    while (ImGuiListClipper_Step(clipper))
      for (int row = clipper->DisplayStart; row < clipper->DisplayEnd; row++) {
        igSelectable_Bool(sort[row].name, false, 0, (ImVec2){0, 0});
        rows++;
      }
    ImGuiListClipper_End(clipper);
    igEnd();
    igRender();
    assert(rows > 0 && rows < 50);
  }
  ImGuiListClipper_destroy(clipper);
  igDestroyContext(ctx);
  free(sort);
  free(names);
}
static void inference_binding(void) {
  char root[]="/tmp/r3d-inference-bind-XXXXXX";assert(mkdtemp(root));
  char a[256],b[256],path[256],heads[32][640];
  snprintf(a,sizeof a,"%s/a.json",root);snprintf(b,sizeof b,"%s/b.json",root);
  FILE *f=fopen(a,"w");assert(f);fclose(f);
  f=fopen(b,"w");assert(f);fclose(f);
  snprintf(path,sizeof path,"%s/ct-manifest.txt",root);
  f=fopen(path,"w");assert(f);char *canonical=realpath(a,NULL);assert(canonical);
  fprintf(f,"%s\n",canonical);fclose(f);free(canonical);
  snprintf(path,sizeof path,"%s/heads.txt",root);
  f=fopen(path,"w");assert(f);fputs("abc/ink\nabc/surface_in1\n",f);fclose(f);
  assert(inference_bound_heads(root,a,heads)==2);
  assert(inference_bound_heads(root,b,heads)==0); /* same format, different CT */
  assert(!inference_request_volume(root,b));
  snprintf(path,sizeof path,"%s/requested-ct.txt",root);
  f=fopen(path,"r");assert(f);char request[1024];assert(fgets(request,sizeof request,f));fclose(f);
  request[strcspn(request,"\r\n")]=0;canonical=realpath(b,NULL);assert(!strcmp(request,canonical));free(canonical);
  assert(inference_bound_heads(root,b,heads)==0); /* request alone is not readiness */
  assert(!inference_request_volume(root,NULL));
  f=fopen(path,"r");assert(f);assert(fgetc(f)=='\n');fclose(f);unlink(path);
  snprintf(path,sizeof path,"%s/heads.txt",root);unlink(path);
  snprintf(path,sizeof path,"%s/ct-manifest.txt",root);unlink(path);
  unlink(a);unlink(b);rmdir(root);
}

static void section_availability(void) {
  ImGuiContext *ctx=igCreateContext(NULL);
  ImGuiIO *io=igGetIO_Nil();io->DisplaySize=(ImVec2){640,480};
  io->DeltaTime=1.0f/60.0f;io->IniFilename=NULL;
  io->BackendFlags|=ImGuiBackendFlags_RendererHasTextures;
  float height=0;
  for(int state=0;state<3;state++) {
    igNewFrame();igSetNextWindowSize((ImVec2){400,300},ImGuiCond_Always);
    igBegin("availability",NULL,0);
    bool enabled=state!=1;
    bool expanded=gui_section("views",enabled,"Open a volume",ImGuiTreeNodeFlags_DefaultOpen);
    ImVec2 size=igGetItemRectSize();
    assert(size.y>0);if(state)assert(size.y==height);height=size.y;
    assert(expanded==enabled); /* disabling preserves expansion for reactivation */
    igEnd();igRender();
  }
  igDestroyContext(ctx);
}

int main(void) {
  inference_binding();
  section_availability();
  corpus();
  clipped_list();
  puts("app corpus leases/sorting/clipping: OK");
  return 0;
}
