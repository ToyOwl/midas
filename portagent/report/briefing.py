"""Briefings from observed price behaviour, without inventing news causality."""
from __future__ import annotations

import math


def percent(value, signed=False):
    return f"{value * 100:{'+' if signed else ''}.1f}%".replace(".", ",")


def asset_brief(row, components, move=None):
    move = move or {}
    one, five, dd = move.get("ret_1d"), move.get("ret_5d"), row.get("drawdown")
    one = one if isinstance(one, (int, float)) and math.isfinite(one) else None
    five = five if isinstance(five, (int, float)) and math.isfinite(five) else None
    dd = dd if isinstance(dd, (int, float)) and math.isfinite(dd) else None
    parts = components or {}
    leading = max(parts, key=parts.get) if parts else None
    if five is None:
        situation = f"Цена на {percent(dd)} ниже недавнего максимума." if dd is not None else "Свежего ценового среза нет."
    elif five < -.015:
        situation = f"За пять сессий потеряла {percent(abs(five))}."
        if one is not None and one > .001:
            situation += f" Последнее движение вверх на {percent(one)} пока отыграло лишь часть снижения."
    elif abs(five) < .01 and one is not None and one < -.001:
        situation = f"За пять сессий почти не сдвинулась ({percent(five, True)}), последнее закрытие — {percent(one, True)}."
    elif five > .015 and dd is not None and dd > .15:
        situation = f"Отскочила на {percent(five)}, но остаётся на {percent(dd)} ниже недавнего максимума."
    elif five > 0:
        situation = f"Прибавила {percent(five)} за пять сессий"
        situation += f", остаётся на {percent(dd)} ниже недавнего максимума." if dd is not None and dd > .1 else "."
    else:
        situation = f"Изменение за пять сессий: {percent(five, True)}."
    if not row.get("warned"):
        importance = "Текущая оценка риска ниже порога предупреждения."
        watch = "Сохранится ли направление движения на следующих закрытиях."
    elif leading == "sentiment" and parts[leading] > .01:
        importance = "Предупреждение усилили улучшившиеся новости, а не ухудшение новостного фона."
        watch = "Сохранится ли рост цены без резких скачков; сам новостной фон здесь не ухудшается."
    elif five is not None and five < -.015 and parts.get("vol_of_vol", 0) > .1:
        importance = "Снижение сочетается с неустойчивым размахом колебаний."
        watch = "Прекратится ли снижение и станет ли восстановление устойчивее одного дня."
    elif parts.get("macro", 0) > .5:
        importance = "Бумага чувствительна к внешним рыночным колебаниям"
        importance += "; небольшой рост не снимает риск нового рывка вниз." if five is not None and five > 0 else "."
        watch = "Не начнёт ли цена снижаться вместе с рынком и расширяться дневной диапазон."
    elif five is not None and five > .015 and parts.get("vol_of_vol", 0) > .5:
        importance = "Восстановление идёт на фоне неровных колебаний — один отскок ещё не закрепляет разворот."
        watch = "Удержится ли прибавка в следующие несколько сессий без резкого отката."
    elif five is not None and five > .015 and parts.get("p_tail", 0) > .1:
        importance = "Отскок заметен, но история резких просадок всё ещё повышает оценку риска."
        watch = "Продолжится ли восстановление или продавцы быстро отыграют недавний рост."
    elif parts.get("earnings", 0) > .01:
        importance = "Приближающаяся отчётность может изменить ожидания по компании."
        watch = "Результаты отчётности и реакция цены после публикации."
    elif parts.get("legal", 0) > .01:
        importance = "В разборе есть судебные претензии, способные повлиять на ожидания инвесторов."
        watch = "Новые сообщения по делу и подтверждённые последствия для компании."
    elif parts.get("p_tail", 0) > .1:
        importance = "История резких падений выделяет бумагу среди остальных позиций."
        watch = "Не сменится ли спокойное движение ускоренным снижением."
    else:
        importance = "Сохранённый индикатор требует более внимательного наблюдения." if row.get("warned") else "Порог предупреждения не пройден."
        watch = "Направление следующих закрытий и размер дневных колебаний."
    return {"situation": situation, "importance": importance, "watch": watch}
