void foo() noexcept{}
void bar() {}
int main() {
  // void (*p)();
  // p = foo;// works
  // void (*p)() noexcept;
  // p = foo; // works
  // void (*p)() noexcept;
  // p = bar; // not compiled
  void (*p)() noexcept;
  p = foo; // works
  return 0;
}