#include <iostream>
class Document {
  public:
    virtual void Serialize() = 0;
};
class Text : public Document {
  public:
    void Serialize() override {std::cout << " from Text\n"; };
};
class XML : virtual public Document {
  public:
    void Serialize() override {std::cout << " from XML\n"; };
};
int main() {
  XML xml;
  xml.Serialize();
  return 0;
}