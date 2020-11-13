/*
 f: any floating point reg
 t: top floating point reg
 u: second floating point reg
 */
#include <stdio.h>

int main() {
  float  angle = 90.;
  float  radian, cosine, sine;
  radian = angle/180.*3.14159;
  __asm__(
    "fsincos"
    :"=t"(cosine),"=u"(sine)
    :"0"(radian)
  );
  printf("The result is %f %f\n", sine, cosine);
  return 0;
}
