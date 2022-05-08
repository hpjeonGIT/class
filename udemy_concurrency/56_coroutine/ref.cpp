#include <coroutine>

#include <iostream>
#include <cassert>


class my_resumable {
public:
   struct promise_type;
   using coro_handle = std::coroutine_handle<promise_type>;
   my_resumable(coro_handle handle) : handle_(handle) { assert(handle); }
   my_resumable(my_resumable&) = delete;
   my_resumable(my_resumable&&) = delete;
   bool resume() {
       if (not handle_.done())
           handle_.resume();
       return not handle_.done();
   }
   ~my_resumable() { handle_.destroy(); }
private:
   coro_handle handle_;
};

struct my_resumable::promise_type {
   using coro_handle = std::coroutine_handle<promise_type>;
   auto get_return_object() noexcept {
       return coro_handle::from_promise(*this);
   }
   auto initial_suspend() noexcept { return std::suspend_always(); }
   auto final_suspend() noexcept{ return std::suspend_always(); }
   void return_void() noexcept {}
   void unhandled_exception() noexcept {
       std::terminate();
   }
};

my_resumable foo() {
   std::cout << "Hello" << std::endl;
   co_await std::suspend_always();
   std::cout << "World" << std::endl;
}

void alpha()
{
   std::cout << "A" << std::endl;
   std::cout << "B" << std::endl;
   std::cout << "C" << std::endl;
}

void numeric()
{
   std::cout << "1" << std::endl;
   std::cout << "2" << std::endl;
   std::cout << "3" << std::endl;
}

int main() 
{
   alpha();
   numeric();
   // resumable res = foo();
   // foo().resume();
   // while (res.resume());
}

