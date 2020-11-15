.text
.globl sum

sum:
  xorpd %xmm0, %xmm0  # initialize sum to zero
  cmp   $0, %rsi       # check if length is zero
  je    done

top:
  addsd (%rdi), %xmm0  # add the current array element
  add   $8, %rdi       # move to next array element
  dec   %rsi
  jnz   top

done:
  ret
