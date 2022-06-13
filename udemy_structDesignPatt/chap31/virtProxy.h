#pragma once
#include <string>
#include <memory>
class Image {
  std::string m_FileName;
protected:
  void SetFileName(const std::string & fileName);
public:
  Image() = default;
  Image(const std::string & fileName);
  const std::string& GetFileName() const;
  virtual ~Image() = default;
  virtual void Display() = 0;
  virtual void Load() = 0;
  virtual void Load(const std::string & fileName) = 0;
};
class Bitmap: public Image {
  std::string m_Buffer{};
public:
  using Image::Image; // c++11 feature. Automatic constructor
  void Display() override;
  void Load() override;
  void Load(const std::string & fileName) override;
};
class BitmapProxy: public Image {
  std::shared_ptr<Bitmap> m_pBitmap{};
  std::string m_FileName;
  bool m_IsLoaded{false};
public:
  BitmapProxy();
  BitmapProxy(const std::string& fileName);
  ~BitmapProxy() = default;
  void Display() override;
  void Load() override;
  void Load(const std::string & fileName) override;
};