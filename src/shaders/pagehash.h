/* Wide virtual block IDs use two uint words: no shaderInt64 requirement on
 * MoltenVK. Entries are {key low, key high, value, padding}, 16 bytes each. */
uint page_hash32(uint x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
uint resident_page64(uint low, uint high) {
  uint mask=page[60], bucket=page_hash32(low ^ page_hash32(high))&mask;
  for(uint probe=0;probe<=mask;probe++,bucket=(bucket+1u)&mask) {
    uint base=64u+bucket*4u, lo=page[base], hi=page[base+1u];
    if(lo==low && hi==high)return page[base+2u];
    if(lo==0xffffffffu && hi==0xffffffffu)break;
  }
  return 0xffffffffu;
}
uint resident_page(uint index) { return resident_page64(index-64u,0u); }
/* High half of a 32x32 product, expressed with portable uint operations. */
uint page_mul_hi(uint a,uint b) {
  uint a0=a&65535u,a1=a>>16,b0=b&65535u,b1=b>>16;
  uint w0=a0*b0,t=a1*b0+(w0>>16),w1=t&65535u,w2=t>>16;
  w1+=a0*b1;
  return a1*b1+w2+(w1>>16);
}
uint resident_page_xyz(uint h,uint x,uint y,uint z,uint dx,uint dy) {
  uint lo=z*dy,hi=page_mul_hi(z,dy),old=lo;
  lo+=y;hi+=uint(lo<old);
  hi=hi*dx+page_mul_hi(lo,dx);lo*=dx;
  old=lo;lo+=x;hi+=uint(lo<old);
  old=lo;lo+=page[h];hi+=page[h+3u]+uint(lo<old);
  return resident_page64(lo,hi);
}
