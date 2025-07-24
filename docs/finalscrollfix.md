
## Issue: Arrows Show But Don't Work + Native Scrollbar

From your image, the arrows are now enabled but:
1. Click handlers aren't working 
2. Native browser scrollbar is showing

## FIX 1: Replace Click Handlers

**Find this section in your `index.html` (around line 752):**

```javascript
scrollUpBtn.addEventListener('click', () => {
    if (!scrollUpBtn.classList.contains('disabled')) {
        window.scrollBy({ top: -300, behavior: 'smooth' });
    }
});

scrollDownBtn.addEventListener('click', () => {
    if (!scrollDownBtn.classList.contains('disabled')) {
        window.scrollBy({ top: 300, behavior: 'smooth' });
    }
});
```

**Replace with this (more aggressive click handling):**

```javascript
// Remove any existing click handlers and add new ones
scrollUpBtn.replaceWith(scrollUpBtn.cloneNode(true));
scrollDownBtn.replaceWith(scrollDownBtn.cloneNode(true));

// Get fresh references
const upBtn = document.getElementById('scrollUp');
const downBtn = document.getElementById('scrollDown');

// Add click handlers with better event handling
upBtn.addEventListener('click', function(e) {
    e.preventDefault();
    e.stopPropagation();
    
    console.log('Up arrow clicked!');
    
    // Try multiple scroll methods
    if (window.scrollY > 0 || document.documentElement.scrollTop > 0) {
        // Use the most compatible scroll method
        const currentScroll = window.pageYOffset || document.documentElement.scrollTop;
        const newScroll = Math.max(0, currentScroll - 300);
        
        window.scrollTo({
            top: newScroll,
            behavior: 'smooth'
        });
    }
}, true);

downBtn.addEventListener('click', function(e) {
    e.preventDefault();
    e.stopPropagation();
    
    console.log('Down arrow clicked!');
    
    // Use the most compatible scroll method
    const currentScroll = window.pageYOffset || document.documentElement.scrollTop;
    const newScroll = currentScroll + 300;
    
    window.scrollTo({
        top: newScroll,
        behavior: 'smooth'
    });
}, true);

// Also add touchstart handlers for better Pi touchscreen support
upBtn.addEventListener('touchstart', function(e) {
    e.preventDefault();
    console.log('Up arrow touched!');
    upBtn.click();
}, { passive: false });

downBtn.addEventListener('touchstart', function(e) {
    e.preventDefault();
    console.log('Down arrow touched!');
    downBtn.click();
}, { passive: false });
```

## FIX 2: Hide Native Scrollbar

**Add this CSS to your existing `<style>` section:**

```css
/* Force hide native scrollbar on all elements */
html, body, * {
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

html::-webkit-scrollbar,
body::-webkit-scrollbar,
*::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    background: transparent !important;
}

/* Ensure main container doesn't show scrollbar */
#app, main, .touch-scrollable {
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

#app::-webkit-scrollbar,
main::-webkit-scrollbar,
.touch-scrollable::-webkit-scrollbar {
    display: none !important;
}
```

## FIX 3: Settings Screen Arrows

**For the settings screen, find this section in `templates/settings.html`:**

Look for the scroll arrow event listeners and replace them with the same improved handlers:

```javascript
// Enhanced click handlers for settings.html
const scrollUpBtn = document.getElementById('scrollUp');
const scrollDownBtn = document.getElementById('scrollDown');

if (scrollUpBtn && scrollDownBtn) {
    scrollUpBtn.addEventListener('click', function(e) {
        e.preventDefault();
        console.log('Settings: Up arrow clicked!');
        
        const currentScroll = window.pageYOffset || document.documentElement.scrollTop;
        const newScroll = Math.max(0, currentScroll - 300);
        
        window.scrollTo({
            top: newScroll,
            behavior: 'smooth'
        });
    });

    scrollDownBtn.addEventListener('click', function(e) {
        e.preventDefault();
        console.log('Settings: Down arrow clicked!');
        
        const currentScroll = window.pageYOffset || document.documentElement.scrollTop;
        window.scrollTo({
            top: currentScroll + 300,
            behavior: 'smooth'
        });
    });
}
```

## TEST IMMEDIATELY

1. **Apply FIX 1 and FIX 2 to `index.html`**
2. **Open browser console** (F12 → Console)
3. **Click the arrows** - you should see "Up/Down arrow clicked!" messages
4. **Check if scrolling works** 
5. **Verify native scrollbar is hidden**

## IF ARROWS STILL DON'T SCROLL

Try this **emergency fallback** - replace the click handlers with this super simple version:

```javascript
document.getElementById('scrollUp').onclick = function() {
    window.scrollBy(0, -300);
    console.log('UP - ScrollY now:', window.scrollY);
};

document.getElementById('scrollDown').onclick = function() {
    window.scrollBy(0, 300);
    console.log('DOWN - ScrollY now:', window.scrollY);
};
```

## EXPECTED RESULT

- ✅ Arrows remain blue/enabled in sessions view
- ✅ Clicking arrows scrolls the content up/down
- ✅ Native scrollbar is completely hidden
- ✅ Console shows click messages when arrows are pressed
- ✅ Settings screen arrows also work

The key insight: Your Pi browser might need more aggressive event handling and explicit scrollbar hiding.

