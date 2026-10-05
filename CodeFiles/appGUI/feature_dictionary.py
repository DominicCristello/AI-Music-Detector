import re
import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from PIL import Image, ImageDraw, ImageTk


def _set_book_icon(window):
    """High-contrast open-book title-bar icon for the dictionary window."""
    s = 4
    image = Image.new('RGBA', (64*s, 64*s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    ink = (130, 210, 255, 255)
    page = (235, 245, 255, 255)
    draw.polygon([(5*s,14*s),(29*s,20*s),(29*s,55*s),(5*s,47*s)], fill=page, outline=ink)
    draw.polygon([(59*s,14*s),(35*s,20*s),(35*s,55*s),(59*s,47*s)], fill=page, outline=ink)
    draw.line([(32*s,19*s),(32*s,56*s)], fill=ink, width=3*s)
    resampling = getattr(Image, 'Resampling', Image).LANCZOS
    icon = ImageTk.PhotoImage(image.resize((64,64), resampling))
    window._book_icon = icon
    if hasattr(window, '_iconbitmap_method_called'):
        window._iconbitmap_method_called = True
    window.iconphoto(False, icon)


def _dark_titlebar(window):
    """Request a dark native title bar and border on supported Windows builds."""
    try:
        import ctypes
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        enabled = ctypes.c_int(1)
        for attribute in (20, 19):
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attribute, ctypes.byref(enabled), ctypes.sizeof(enabled)) == 0:
                break
    except Exception:
        pass


def _glow_image(width, height, rgb, strength):
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    alpha = np.clip(1.0 - np.sqrt(xx*xx + yy*yy), 0, 1) ** 2.4
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = np.clip(alpha * strength, 0, 255).astype(np.uint8)
    return ImageTk.PhotoImage(Image.fromarray(rgba))


def _meaning(name):
    n = name.lower()
    families = [
        ('rms', 'signal energy and perceived loudness'),
        ('spectralcontrast', 'the separation between spectral peaks and valleys'),
        ('spectralflatness', 'how noise-like rather than tonal the spectrum is'),
        ('spectralrolloff', 'the frequency below which most spectral energy lies'),
        ('spectralcentroid', 'the spectrum’s brightness or centre of mass'),
        ('spectralflux', 'the rate of frame-to-frame spectral change'),
        ('chroma', 'the distribution of energy among the twelve pitch classes'),
        ('zcr', 'the rate at which the waveform crosses zero'),
        ('tempo', 'the estimated speed or stability of the musical pulse'),
        ('onset', 'the timing and regularity of detected musical attacks'),
        ('pitch', 'the stability or variation of estimated pitch'),
        ('correlation', 'the similarity and phase agreement between channels'),
        ('harmonic', 'the strength and organization of harmonic components'),
        ('phase', 'the consistency of phase relationships in the signal'),
        ('rhythm', 'the variation and structural complexity of rhythmic events'),
        ('duration', 'the total running time of the analyzed recording'),
        ('sample', 'the digital sampling rate used by the recording'),
    ]
    concept = next((v for k, v in families if k in n),
                   'the named statistical property of the analyzed audio')
    stat = ('variability' if any(x in n for x in ('std', 'variance', 'jitter'))
            else 'average magnitude')
    return concept, stat


def describe(name, rule=None):
    concept, stat = _meaning(name)
    first = f'This parameter reports the {stat} of {concept}.'
    second = ('A high value indicates a strong or highly variable manifestation; '
              'a low value indicates a weak or tightly controlled manifestation; '
              'a mid-range value indicates moderate behaviour.')
    if rule:
        ai_high = rule[2]
        third = (('In this detector, unusually high values are associated with the AI class, while lower values provide comparatively human evidence.'
                  if ai_high else
                  'In this detector, unusually low values are associated with the AI class, while higher values provide comparatively human evidence.'))
    else:
        third = ('No single value proves authorship; the model determines AI likelihood from this measurement’s learned interaction with the other parameters.')
    return f'{first} {second} {third}'


class FeatureDictionaryWindow:
    def __init__(self, master, terms, rules, icon_setter=None):
        self.terms = list(terms)
        self.rules = rules
        self._resize_job = None
        self._last_size = None
        self.win = tk.Toplevel(master)
        self.win.title('Track Analysis Dictionary')
        self.win.geometry('1280x820')
        self.win.minsize(900, 620)
        self.win.configure(bg='#050512')
        _set_book_icon(self.win)
        _dark_titlebar(self.win)
        self.canvas=tk.Canvas(self.win,bg='#050512',highlightthickness=0)
        sb=tk.Scrollbar(self.win,command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right',fill='y')
        self.canvas.pack(fill='both',expand=True)
        self.win.update_idletasks()
        self._render()
        self.win.bind('<Configure>', self._schedule_render)
        self.win.bind('<MouseWheel>',
                      lambda e:self.canvas.yview_scroll(int(-e.delta/120),'units'))
        self.win.lift(); self.win.attributes('-topmost',True); self.win.focus_force()
        self.win.after(300,lambda:self.win.attributes('-topmost',False))

    def _schedule_render(self, event):
        if event.widget is not self.win:
            return
        size = (event.width, event.height)
        if size == self._last_size:
            return
        if self._resize_job is not None:
            try: self.win.after_cancel(self._resize_job)
            except tk.TclError: pass
        self._resize_job = self.win.after(100, self._render)

    def _render(self):
        self._resize_job = None
        self.win.update_idletasks()
        w = max(880, self.canvas.winfo_width())
        h_view = max(600, self.canvas.winfo_height())
        size = (w, h_view)
        if size == self._last_size:
            return
        old_position = self.canvas.yview()[0] if self._last_size else 0.0
        self._last_size = size
        c = self.canvas
        c.delete('all')
        c.create_text(w//2,48,text='Track Analysis Dictionary',fill='white',
                      font=('Segoe UI',24,'bold'), tags='content')

        y = 105
        margin = max(32, int(w * .03))
        term_x = margin
        term_size = 10
        max_term_width = max(330, min(610, w - (margin * 2) - 385))
        term_font = tkfont.Font(family='Consolas', size=term_size)
        longest = max((term_font.measure(name) for name in self.terms), default=300)
        while longest + 24 > max_term_width and term_size > 7:
            term_size -= 1
            term_font.configure(size=term_size)
            longest = max(term_font.measure(name) for name in self.terms)
        term_width = min(max_term_width, longest + 24)
        line_x1 = term_x + term_width + 20
        line_x2 = line_x1 + 32
        description_x = line_x2 + 20
        description_width = max(280, w - description_x - margin - 18)
        for name in self.terms:
            term_id = c.create_text(
                term_x + 12, y, text=name, fill='white', anchor='nw',
                font=term_font, tags=('content', 'term_text'))
            description_id = c.create_text(
                description_x, y, text=describe(name,self.rules.get(name)),
                fill='white', anchor='nw', width=description_width,
                font=('Segoe UI',11), justify='left', tags='content')
            c.update_idletasks()
            term_box = c.bbox(term_id)
            description_box = c.bbox(description_id)
            term_height = term_box[3] - term_box[1]
            description_height = description_box[3] - description_box[1]
            row_height = max(term_height, description_height, 36)
            c.create_rectangle(
                term_x, y - 7, term_x + term_width, y + term_height + 7,
                fill='black', outline='black', tags='term_background')
            c.tag_lower('term_background', 'content')
            c.create_line(line_x1, y + 14, line_x2, y + 14,
                          fill='white', width=1, tags='content')
            y += row_height + 92

        content_height = y + 35
        c.configure(scrollregion=(0,0,w,content_height))

        # Sparse, non-repeating backdrop: one subtle blue glow and only three
        # small magenta blotches across the complete dictionary.
        self._glows = [
            _glow_image(min(720,w), 480, (20,70,170), 55),
            _glow_image(min(470,w), 330, (170,20,145), 72),
        ]
        c.create_image(w*.78, 160, image=self._glows[0], tags='gradient')
        for gx, gy in ((w*.18, content_height*.22),
                       (w*.82, content_height*.57),
                       (w*.28, content_height*.86)):
            c.create_image(gx, gy, image=self._glows[1], tags='gradient')
        c.tag_lower('gradient')
        c.yview_moveto(old_position)
