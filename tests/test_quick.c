/* CPU-only quick tests (ctest label: quick). No window, no GPU. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "core/mathx.h"

static int failures = 0;
#define CHECK(cond)                                                  \
  do {                                                               \
    if (!(cond)) {                                                   \
      failures++;                                                    \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                \
  } while (0)

static int feq(float a, float b) { return fabsf(a - b) < 1e-6f; }

static void test_mathx(void) {
  r3d_v3 a = v3(1, 2, 3), b = v3(4, 5, 6);
  CHECK(feq(v3_dot(a, b), 32.0f));
  r3d_v3 c = v3_cross(v3(1, 0, 0), v3(0, 1, 0));
  CHECK(feq(c.x, 0) && feq(c.y, 0) && feq(c.z, 1));
  CHECK(feq(v3_len(v3(3, 4, 0)), 5.0f));
  r3d_v3 n = v3_norm(v3(0, 0, 9));
  CHECK(feq(n.z, 1.0f));
  CHECK(feq(v3_len(v3_norm(v3(0, 0, 0))), 0.0f)); /* zero vector stays zero */
  CHECK(feq(fclampf(5, 0, 1), 1.0f));
  CHECK(feq(flerpf(2, 4, 0.5f), 3.0f));
  r3d_v3 s = v3_add(v3_scale(a, 2.0f), v3_sub(b, a));
  CHECK(feq(s.x, 5) && feq(s.y, 7) && feq(s.z, 9));
}

int main(void) {
  test_mathx();
  if (failures) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return EXIT_FAILURE;
  }
  printf("test_quick: all ok\n");
  return EXIT_SUCCESS;
}
