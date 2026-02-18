use iced::widget::{container, rich_text, row, span, stack, text, Space};
use iced::{Color, Element, Theme};

use crate::animations::PulseAnimation;

// ── Layout (4px grid) ──

const PILL_WIDTH: f32 = 500.0;
const PILL_RADIUS: f32 = 20.0;
const DOT_SIZE: f32 = 8.0;
const DOT_GLOW_SIZE: f32 = 20.0;
const DOT_GLOW_PAD: f32 = (DOT_GLOW_SIZE - DOT_SIZE) / 2.0;

// ── Accent ──

const ACCENT: Color = Color {
    r: 0.30,
    g: 0.52,
    b: 1.0,
    a: 1.0,
};

// ── Light theme ──

const LIGHT_BG: [f32; 4] = [0.98, 0.98, 1.0, 0.95];

const LIGHT_BORDER: Color = Color {
    r: 0.0,
    g: 0.0,
    b: 0.12,
    a: 0.07,
};

const LIGHT_TEXT: Color = Color {
    r: 0.07,
    g: 0.07,
    b: 0.09,
    a: 1.0,
};

const LIGHT_TEXT_MUTED: Color = Color {
    r: 0.38,
    g: 0.38,
    b: 0.44,
    a: 1.0,
};

// ── Dark theme ──

const DARK_BG: [f32; 4] = [0.10, 0.10, 0.14, 0.93];

const DARK_BORDER: Color = Color {
    r: 0.42,
    g: 0.42,
    b: 0.58,
    a: 0.14,
};

const DARK_TEXT: Color = Color {
    r: 0.93,
    g: 0.93,
    b: 0.96,
    a: 1.0,
};

const DARK_TEXT_MUTED: Color = Color {
    r: 0.48,
    g: 0.48,
    b: 0.55,
    a: 1.0,
};

// ── Styles ──

fn pill_style(opacity: f32, dark: bool) -> impl Fn(&Theme) -> container::Style {
    let bg = if dark { DARK_BG } else { LIGHT_BG };
    let fg = if dark { DARK_TEXT } else { LIGHT_TEXT };
    let border = if dark { DARK_BORDER } else { LIGHT_BORDER };

    move |_theme: &Theme| container::Style {
        background: Some(iced::Background::Color(Color::from_rgba(
            bg[0], bg[1], bg[2], bg[3] * opacity,
        ))),
        border: iced::Border {
            radius: PILL_RADIUS.into(),
            width: 1.0,
            color: Color { a: border.a * opacity, ..border },
        },
        shadow: iced::Shadow::default(),
        text_color: Some(Color { a: fg.a * opacity, ..fg }),
        snap: false,
    }
}

/// Pulsing dot with a soft glow halo behind it.
fn build_dot<'a, Message: 'a>(pulse_op: f32, opacity: f32) -> Element<'a, Message> {
    let glow_alpha = pulse_op * opacity * 0.18;
    let dot_alpha = pulse_op * opacity;

    let glow = container(Space::new())
        .width(DOT_GLOW_SIZE)
        .height(DOT_GLOW_SIZE)
        .style(move |_: &Theme| container::Style {
            background: Some(iced::Background::Color(Color {
                a: glow_alpha,
                ..ACCENT
            })),
            border: iced::Border {
                radius: (DOT_GLOW_SIZE / 2.0).into(),
                ..Default::default()
            },
            ..Default::default()
        });

    let dot = container(
        container(Space::new())
            .width(DOT_SIZE)
            .height(DOT_SIZE)
            .style(move |_: &Theme| container::Style {
                background: Some(iced::Background::Color(Color {
                    a: dot_alpha,
                    ..ACCENT
                })),
                border: iced::Border {
                    radius: (DOT_SIZE / 2.0).into(),
                    ..Default::default()
                },
                ..Default::default()
            }),
    )
    .padding(DOT_GLOW_PAD as u16)
    .width(DOT_GLOW_SIZE)
    .height(DOT_GLOW_SIZE);

    stack![glow, dot].into()
}

// ── Public API ──

pub fn overlay_pill<'a, Message: 'a>(
    words: &[(String, f32)],
    pulse: &PulseAnimation,
    opacity: f32,
    dark_mode: bool,
) -> Element<'a, Message> {
    let text_color = if dark_mode { DARK_TEXT } else { LIGHT_TEXT };
    let muted = if dark_mode { DARK_TEXT_MUTED } else { LIGHT_TEXT_MUTED };

    let dot = build_dot::<Message>(pulse.opacity(), opacity);

    if words.is_empty() {
        let text_pulse = 0.7 + 0.3 * pulse.opacity();
        let label = text("listening")
            .size(13)
            .color(Color {
                a: muted.a * opacity * text_pulse,
                ..muted
            });

        let inner = row![dot, label]
            .spacing(12)
            .align_y(iced::Alignment::Center);

        container(inner)
            .padding([10, 18])
            .width(PILL_WIDTH)
            .style(pill_style(opacity, dark_mode))
            .into()
    } else {
        const MAX_WORDS: usize = 80;

        let (visible, truncated) = if words.len() > MAX_WORDS {
            (&words[words.len() - MAX_WORDS..], true)
        } else {
            (words, false)
        };

        let mut spans: Vec<iced::widget::text::Span<'_, ()>> = Vec::with_capacity(visible.len() + 1);

        if truncated {
            spans.push(span("... ").color(Color {
                a: muted.a * opacity * 0.6,
                ..muted
            }));
        }

        for (i, (word, word_opacity)) in visible.iter().enumerate() {
            let s = if i < visible.len() - 1 {
                format!("{} ", word)
            } else {
                word.clone()
            };
            spans.push(span(s).color(Color {
                a: text_color.a * opacity * word_opacity,
                ..text_color
            }));
        }

        let label = rich_text(spans).size(14);

        let inner = row![dot, label]
            .spacing(12)
            .align_y(iced::Alignment::Center);

        container(inner)
            .padding([10, 18])
            .width(PILL_WIDTH)
            .style(pill_style(opacity, dark_mode))
            .into()
    }
}
