"""
bot/handlers/products.py
------------------------
Manages the product catalog for the management bot.

Features:
  - '📦 Our Products' button → catalog summary with brand buttons.
  - Brand button → category list for that brand.
  - Category button → full product list with prices.
  - Any .txt document upload → parses the price list, REPLACES all products.
  - Inline '🗑 Clear all' button with confirmation step.
"""

import asyncio
import io

from aiogram import F, Router
from aiogram.exceptions import TelegramNetworkError
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from bot.keyboards.main_menu import BTN_PRODUCTS
from data.products import clear_and_replace, get_catalog_summary, load_products, validate_catalog_coverage
from utils.logger import get_logger
from utils.txt_parser import parse_txt

logger = get_logger(__name__)
router = Router()

_CB_CLEAR         = "products_clear"
_CB_CLEAR_CONFIRM = "products_clear_confirm"
_CB_CLEAR_CANCEL  = "products_cancel"
_CB_BACK          = "prod_back"
_CB_BRAND_PFX     = "prod_br:"      # prod_br:{brand}
_CB_CAT_PFX       = "prod_cat:"     # prod_cat:{brand}|{category}
_CB_BACK_BR_PFX   = "prod_backbr:"  # prod_backbr:{brand}


# ─── Text builders ────────────────────────────────────────────────────────────

def _summary_text(summary: dict) -> str:
    total = summary["total"]
    if total == 0:
        return (
            "📦 <b>Наши товары</b>\n\n"
            "Товары пока не загружены.\n\n"
            "Отправьте в этот чат файл прайса <code>.txt</code> для загрузки."
        )
    lines = [f"📦 <b>Наши товары</b> — {total} позиций\n"]
    for brand, categories in summary["brands"].items():
        brand_total = sum(categories.values())
        lines.append(f"\n<b>{brand}</b> ({brand_total})")
        for cat, count in categories.items():
            lines.append(f"  {cat}: {count}")
    lines.append(
        "\n\n<i>Нажмите на бренд для просмотра. Для обновления отправьте файл <code>.txt</code>.</i>"
    )
    return "\n".join(lines)


def _summary_keyboard(summary: dict) -> InlineKeyboardMarkup:
    rows = []
    for brand, categories in summary["brands"].items():
        brand_total = sum(categories.values())
        rows.append([InlineKeyboardButton(
            text=f"{brand}  ({brand_total})",
            callback_data=f"{_CB_BRAND_PFX}{brand}",
        )])
    if summary["total"] > 0:
        rows.append([InlineKeyboardButton(text="🗑 Очистить все", callback_data=_CB_CLEAR)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _brand_text(brand: str, categories: dict) -> str:
    total = sum(categories.values())
    lines = [f"<b>{brand}</b> — {total} позиций\n"]
    for cat, count in categories.items():
        lines.append(f"  {cat}: {count}")
    return "\n".join(lines)


def _brand_keyboard(brand: str, categories: dict) -> InlineKeyboardMarkup:
    rows = []
    for cat, count in categories.items():
        rows.append([InlineKeyboardButton(
            text=f"{cat}  ({count})",
            callback_data=f"{_CB_CAT_PFX}{brand}|{cat}",
        )])
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data=_CB_BACK)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _category_text(brand: str, category: str, products: list) -> str:
    lines = [f"<b>{brand} — {category}</b> ({len(products)} позиций)\n"]
    for p in products:
        name  = p.get("name", "—")
        price = p.get("price", "—")
        avail = "🟢" if p.get("available") else "🔴"
        lines.append(f"{avail} {name} — <b>{price}</b>")
    return "\n".join(lines)


def _category_keyboard(brand: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="⬅️ Назад", callback_data=f"{_CB_BACK_BR_PFX}{brand}"),
    ]])


# ─── Handlers ─────────────────────────────────────────────────────────────────

@router.message(F.text == BTN_PRODUCTS)
async def show_products(message: Message) -> None:
    summary = get_catalog_summary()
    for attempt in range(3):
        try:
            await message.answer(
                _summary_text(summary),
                reply_markup=_summary_keyboard(summary),
            )
            return
        except TelegramNetworkError as exc:
            if attempt == 2:
                logger.error("show_products: failed after 3 attempts: %s", exc)
                raise
            logger.warning("show_products: timeout, retrying (%d/3)...", attempt + 1)
            await asyncio.sleep(2)


@router.callback_query(F.data.startswith(_CB_BRAND_PFX))
async def show_brand(callback: CallbackQuery) -> None:
    """Show category list for a brand."""
    brand = callback.data[len(_CB_BRAND_PFX):]
    summary = get_catalog_summary()
    categories = summary["brands"].get(brand, {})
    await callback.message.edit_text(
        _brand_text(brand, categories),
        reply_markup=_brand_keyboard(brand, categories),
    )
    await callback.answer()


@router.callback_query(F.data.startswith(_CB_CAT_PFX))
async def show_category(callback: CallbackQuery) -> None:
    """Show product list for a brand+category."""
    payload = callback.data[len(_CB_CAT_PFX):]
    brand, _, category = payload.partition("|")
    all_products = load_products()
    filtered = [
        p for p in all_products
        if p.get("brand", "") == brand and p.get("category", "") == category
    ]
    text = _category_text(brand, category, filtered)
    # Telegram message limit safety
    if len(text) > 4000:
        text = text[:3950] + "\n\n<i>... (обрезано)</i>"
    await callback.message.edit_text(text, reply_markup=_category_keyboard(brand))
    await callback.answer()


@router.callback_query(F.data == _CB_BACK)
async def back_to_summary(callback: CallbackQuery) -> None:
    """Back to main summary from brand view."""
    summary = get_catalog_summary()
    await callback.message.edit_text(
        _summary_text(summary),
        reply_markup=_summary_keyboard(summary),
    )
    await callback.answer()


@router.callback_query(F.data.startswith(_CB_BACK_BR_PFX))
async def back_to_brand(callback: CallbackQuery) -> None:
    """Back to brand view from category view."""
    brand = callback.data[len(_CB_BACK_BR_PFX):]
    summary = get_catalog_summary()
    categories = summary["brands"].get(brand, {})
    await callback.message.edit_text(
        _brand_text(brand, categories),
        reply_markup=_brand_keyboard(brand, categories),
    )
    await callback.answer()


@router.callback_query(F.data == _CB_CLEAR)
async def confirm_clear(callback: CallbackQuery) -> None:
    await callback.message.edit_reply_markup(
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="✅ Да, удалить все", callback_data=_CB_CLEAR_CONFIRM),
            InlineKeyboardButton(text="❌ Отмена",          callback_data=_CB_CLEAR_CANCEL),
        ]])
    )
    await callback.answer()


@router.callback_query(F.data == _CB_CLEAR_CONFIRM)
async def do_clear(callback: CallbackQuery) -> None:
    clear_and_replace([])
    await callback.message.edit_text("🗑 Все товары удалены.")
    await callback.answer("Готово!")
    logger.info("Admin cleared the entire product catalog.")


@router.callback_query(F.data == _CB_CLEAR_CANCEL)
async def cancel_clear(callback: CallbackQuery) -> None:
    summary = get_catalog_summary()
    await callback.message.edit_text(
        _summary_text(summary),
        reply_markup=_summary_keyboard(summary),
    )
    await callback.answer("Отменено.")


@router.message(F.document)
async def handle_document(message: Message) -> None:
    """
    Handle document uploads.

    Only .txt files are accepted as price lists.
    On success the entire catalog is REPLACED with the parsed content.
    """
    doc = message.document

    if not doc.file_name or not doc.file_name.lower().endswith(".txt"):
        await message.reply(
            "⚠️ Для загрузки прайса отправьте файл <code>.txt</code>."
        )
        return

    await message.reply("⏳ Разбираю прайс, подождите...")

    # Download file into memory
    bio = io.BytesIO()
    await message.bot.download(doc.file_id, destination=bio)
    bio.seek(0)

    try:
        content = bio.read().decode("utf-8")
    except UnicodeDecodeError:
        bio.seek(0)
        content = bio.read().decode("cp1251", errors="replace")

    all_products, total_in_file, unrecognized = parse_txt(content)

    if total_in_file == 0:
        await message.reply(
            "❌ В файле не найдено товаров.\n"
            "Проверьте формат:\n"
            "<code>[Бренд][Категория][Подкатегория][Вариант]</code>\n"
            "<code>Название товара|Цена</code>"
        )
        return

    # ✅ Only keep products marked with 🟢 (available=True)
    available_products = [p for p in all_products if p.get("available")]

    if not available_products:
        await message.reply(
            "⚠️ В файле нет товаров с отметкой 🟢.\n"
            "В каталог добавляются только строки, начинающиеся с <b>🟢</b>."
        )
        return

    clear_and_replace(available_products)
    saved = len(available_products)

    # ── Alias coverage check ──────────────────────────────────────────────────
    uncovered = validate_catalog_coverage()
    coverage_text = ""
    if uncovered:
        sample = uncovered[:10]
        more = len(uncovered) - len(sample)
        lines = "\n".join(f"  • {n}" for n in sample)
        if more > 0:
            lines += f"\n  … and {more} more"
        coverage_text = (
            f"\n\n⚠️ <b>{len(uncovered)} товар(ов) не покрыты алиасами</b> "
            f"(покупатели их не найдут):\n<code>{lines}</code>"
        )

    await message.reply(
        f"✅ <b>Прайс загружен!</b>\n\n"
        f"📄 Всего строк в файле: <b>{total_in_file}</b>\n"
        f"🟢 Добавлено в каталог (только 🟢): <b>{saved}</b>\n\n"
        f"<i>Все предыдущие товары были заменены.</i>"
        f"{coverage_text}"
    )
    logger.info(
        "Price list uploaded: %d total, %d with 🟢 saved.",
        total_in_file, saved,
    )
