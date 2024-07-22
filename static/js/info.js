document.addEventListener('DOMContentLoaded', function() {
    const carousel = document.querySelector('.carousel');
    const slides = document.querySelectorAll('.slide');
    let index = 0;

    function showNextSlide() {
        index = (index + 1) % slides.length;
        carousel.style.transform = `translateX(-${index * 100}%)`;
    }

    setInterval(showNextSlide, 5000);
});

