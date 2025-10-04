# XRP Signal Bot

ربات تحلیل تکنیکال برای ارسال سیگنال خرید/فروش جفت‌های ارز دیجیتال (XRP، BTC، ETH، SOL) در تلگرام. داده‌ها از سرویس CryptoCompare خوانده می‌شوند و نتیجه به مشترکینی که قبلاً در بات شما `/start` زده‌اند ارسال می‌گردد.

## امکانات
- محاسبه RSI، MACD و بررسی واگرایی روی تایم‌فریم‌های روزانه و ۴ ساعته.
- تخمین احتمال موفقیت سیگنال با بک‌تست کوتاه‌مدت.
- ارسال پیام HTML به تلگرام با متن فارسی و انگلیسی.
- اسکریپت کمکی برای ثبت مشترکین جدید (`listen_start.py`).

## پیش‌نیازها
- Python 3.10 یا جدیدتر
- دسترسی به اینترنت برای تماس با API تلگرام و CryptoCompare
- یک ربات تلگرام و توکن آن (از طریق BotFather)
- شناسه چت مقصد (برای گروه یا چت خصوصی)
- کلید API سرویس CryptoCompare (رایگان است، اما اختیاری است؛ بدون آن محدودیت نرخ بیشتری خواهید داشت)

## راه‌اندازی سریع
```bash
# ۱) دریافت پروژه
 git clone https://github.com/<your-username>/xrp-signal-bot.git
 cd xrp-signal-bot

# ۲) ساخت محیط مجزا
 python -m venv .venv
 source .venv/bin/activate   # در ویندوز: .venv\Scripts\activate

# ۳) نصب وابستگی‌ها
 pip install -r requirements.txt

# ۴) ایجاد فایل تنظیمات
 cp .env.example .env
```

### تنظیم `.env`
مقادیر زیر را در فایل `.env` تکمیل کنید:

```
TELEGRAM_BOT_TOKEN=توکن-ربات-تلگرام
TELEGRAM_CHAT_ID=شناسه-چت-یا-گروه (اختیاری، برای fallback)
CRYPTOCOMPARE_API_KEY=کلید-API-کریپتو-کامپر (اختیاری)
# SUBSCRIBERS_PATH=/مسیر/جایگزین/برای/subscribers.json (اختیاری)
```

> نکته: اگر متغیر `ENV_FILE` را در محیط تعریف کنید، همان مسیر برای خواندن `.env` استفاده می‌شود. در غیر این صورت، برنامه ابتدا به دنبال `.env` کنار اسکریپت‌ها و سپس مسیر `~/xrpbot/.env` می‌گردد.

### تست اتصال تلگرام
پس از پر کردن `.env` بهتر است یک پیام آزمایشی بفرستید:
```bash
python send_test.py
```
اگر وضعیت `200` بود و پیام دریافت شد، توکن و chat id صحیح هستند.

### مدیریت مشترکین
اسکریپت `listen_start.py` یا `listen_updates.py` پیام‌های `/start` را می‌خوانند و شناسه چت را در فایل `subscribers.json` ذخیره می‌کنند.

```bash
python listen_start.py
```

- فایل‌های `subscribers.json` و `offset.json` در ریشه ریپو ذخیره می‌شوند. آن‌ها را خصوصی نگه دارید.
- اگر این فایل حداقل یک شناسه داشته باشد، ربات به همه آن‌ها پیام می‌فرستد و نبود متغیر `TELEGRAM_CHAT_ID` مشکلی ایجاد نمی‌کند.
- در صورت نیاز می‌توانید مسیر دیگری را با متغیر `SUBSCRIBERS_PATH` تعیین کنید.
- هر دو اسکریپت هنگام اجرا به صورت خودکار به دنبال `.env` می‌گردند (مانند `trigger_xrp_bot.py`).

### اجرای ربات سیگنال
```bash
python trigger_xrp_bot.py
```

- ابتدا مشترکین را از `subscribers.json` می‌خواند؛ اگر فهرست خالی باشد، از مقدار `TELEGRAM_CHAT_ID` به عنوان fallback استفاده می‌کند.
- در حالت پیش‌فرض (`SEND_ONLY_ON_TRIGGER = True`) فقط زمانی پیام ارسال می‌شود که سیگنال تازه‌ای (خرید یا فروش) تشخیص داده شود.
- برای دریافت گزارش هر بار اجرای ربات، مقدار بالا را به `False` تغییر دهید.

### زمان‌بندی اجرا
- **Cron (سرور شخصی):**
  ```cron
  */30 * * * * /path/to/venv/bin/python /path/to/repo/trigger_xrp_bot.py >> /var/log/xrpbot.log 2>&1
  ```
- **GitHub Actions:** فایل `.github/workflows/xrpbot.yml` اجرای خودکار هر ۵ دقیقه (یا اجرای دستی) را فراهم می‌کند. تنها کافی است secrets (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `CRYPTOCOMPARE_API_KEY`) را در تنظیمات ریپو بسازید.

## ساختار فایل‌ها
- `trigger_xrp_bot.py`: منطق تحلیل و ارسال پیام
- `listen_start.py`, `listen_updates.py`: مدیریت مشترکین تلگرامی
- `send_test.py`: تست سریع ارسال پیام
- `requirements.txt`: وابستگی‌های پایتونی پروژه
- `.github/workflows/xrpbot.yml`: اجرای اصلی ربات روی GitHub Actions
- `.github/workflows/subscriber-listener.yml`: شنود پیام‌های `/start` و بروزرسانی فایل مشترکین

## نکات امنیتی
- فایل `.env` و `subscribers.json` شامل اطلاعات حساس هستند؛ در ریپوی عمومی منتشرشان نکنید.
- برای انتشار، مطمئن شوید که فولدر `bin/` و `lib/` (تولید شده توسط virtualenv) در `.gitignore` باشد.

## مشارکت
پیشنهاد یا باگ جدید دارید؟ Issue بزنید یا Pull Request بفرستید. لطفاً قبل از ارسال، کد را با `python -m compileall .` یا تست‌های خودتان بررسی کنید.
