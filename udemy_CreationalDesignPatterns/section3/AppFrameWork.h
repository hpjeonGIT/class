#pragma once
#include <memory>
class Document {
public:
  virtual void Write() = 0;
  virtual void Read() = 0;
  virtual ~Document() = default;
};
class TextDocument: public Document {
public:
  void Write() override;
  void Read() override;
};
using DocumentPtr = std::unique_ptr<Document>;
class Application {
protected:
  DocumentPtr m_pDocument;
public:
  void New();
  void Open();
  void Save();
  virtual DocumentPtr Create() {return nullptr;}
};
class TextApplication: public Application {
public:
  void New();
  void Open();
  void Save();
  DocumentPtr Create();
};
class SpreadsheetDocument: public Document {
public:
  void Write() override;
  void Read() override;
};
class SpreadsheetApplication: public Application {
public:
  void New();
  void Open();
  void Save();
  DocumentPtr Create();
};