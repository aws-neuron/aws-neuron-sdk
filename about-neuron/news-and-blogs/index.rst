.. meta::
    :description: Links to external news and blog articles about AWS Neuron and Trainium/Inferentia ML accelerators.
    :date-modified: 02/26/2026

.. _neuron-news:

AWS Neuron News and Blogs
=========================

Stay up to date with the latest news, announcements, and technical blog posts about AWS Neuron, AWS Trainium, and AWS Inferentia. Discover customer success stories, performance benchmarks, best practices, and deep dives into machine learning acceleration on AWS.

----

Featured Articles
-----------------

Read recent blogs and technical content about Neuron, Trainium, and Inferentia from AWS subject matter experts and our highly experienced customers.

.. datatemplate:yaml:: news-and-blogs.yaml

   .. grid:: 1
      :gutter: 2
   {% for article in data.featured_articles %}
   {% if article.locale == 'en-US' %}{% set flag = '🇺🇸' %}{% set locale_name = 'English' %}{% elif article.locale == 'ja-JP' %}{% set flag = '🇯🇵' %}{% set locale_name = 'Japanese' %}{% elif article.locale == 'zh-CN' %}{% set flag = '🇨🇳' %}{% set locale_name = 'Chinese' %}{% elif article.locale == 'ko-KR' %}{% set flag = '🇰🇷' %}{% set locale_name = 'Korean' %}{% else %}{% set flag = '🌐' %}{% set locale_name = 'Unknown' %}{% endif %}

      .. grid-item-card::
         :class-card: sd-border-2
         :link: {{ article.url }}

         {{ article.icon }} **{{ article.title }}**
         ^^^
         {{ article.description }}
         +++
         **Published on**: {{ article.date }} | {{ flag }} ({{ locale_name }}) | Content by `{{ article.author }} <{{ article.author_url }}>`__
   {% endfor %}

.. note::
   
   This page is regularly updated with new content. Bookmark it to stay informed about the latest developments in AWS Neuron, Trainium, and Inferentia.
 
**For the full list of featured articles and posts, go to the :ref:`News & Blogs <all-articles>` section of this page.**

.. _all-articles:

News & Blogs 
-------------

Explore the latest news, press releases, and industry coverage about AWS Neuron, Trainium, and Inferentia.

.. raw:: html

   <div style="margin-bottom: 20px;">
     <label for="locale-filter" style="font-weight: bold; margin-right: 10px;">Filter by language:</label>
     <select id="locale-filter" style="padding: 5px 10px; font-size: 14px; border: 1px solid #ccc; border-radius: 4px;">
       <option value="en-US" selected>🇺🇸 English</option>
       <option value="ja-JP">🇯🇵 Japanese</option>
       <option value="ko-KR">🇰🇷 Korean</option>
       <option value="zh-CN">🇨🇳 Chinese</option>
       <option value="all">All languages</option>
     </select>
   </div>

.. datatemplate:yaml:: news-and-blogs.yaml

   .. grid:: 1 1 2 2
      :gutter: 2
      :class-container: articles-grid news-blogs-grid
   {% for article in data.all_articles|sort(attribute='date', reverse=True) %}
   {% if article.locale == 'en-US' %}{% set flag = '🇺🇸' %}{% set locale_name = 'English' %}{% elif article.locale == 'ja-JP' %}{% set flag = '🇯🇵' %}{% set locale_name = 'Japanese' %}{% elif article.locale == 'zh-CN' %}{% set flag = '🇨🇳' %}{% set locale_name = 'Chinese' %}{% elif article.locale == 'ko-KR' %}{% set flag = '🇰🇷' %}{% set locale_name = 'Korean' %}{% else %}{% set flag = '🌐' %}{% set locale_name = 'Unknown' %}{% endif %}

      .. grid-item-card::
         :link: {{ article.url }}
         :class-card: sd-border-1 article-card
         :class-body: article-locale-{{ article.locale }}

         **{{ article.title }}**
         ^^^
         {{ article.description }}
         +++
         **Published on**: {{ article.date }} | {{ flag }} ({{ locale_name }})
   {% endfor %}

.. raw:: html

   <script>
   (function() {
     'use strict';
     
     function initFilter() {
       const filter = document.getElementById('locale-filter');
       const articlesGrid = document.querySelector('.news-blogs-grid');
       
       if (!filter) {
         console.error('Filter dropdown not found!');
         return;
       }
       
       if (!articlesGrid) {
         console.error('News & Blogs grid not found!');
         return;
       }
       
       console.log('Filter and News & Blogs grid found successfully');
       
       // Get all article cards
       const articleCards = Array.from(articlesGrid.querySelectorAll('.sd-col'));
       console.log('Total article cards found:', articleCards.length);
       
       // Extract locale from each card
       const cardLocales = articleCards.map((card, index) => {
         const body = card.querySelector('[class*="article-locale-"]');
         if (!body) {
           console.warn('Card', index, 'has no locale class');
           return 'UNKNOWN';
         }
         
         const classes = body.className.split(' ');
         const localeClass = classes.find(c => c.startsWith('article-locale-'));
         
         if (!localeClass) {
           console.warn('Card', index, 'has no article-locale- class');
           return 'UNKNOWN';
         }
         
         // Convert "article-locale-ja-jp" to "JA-JP"
         const locale = localeClass.replace('article-locale-', '').toUpperCase();
         console.log('Card', index, 'locale:', locale);
         return locale;
       });
       
       // Function to apply filter
       function applyFilter(selectedLocale) {
         console.log('=== Applying filter:', selectedLocale, '===');
         
         let visibleCount = 0;
         
         articleCards.forEach((card, index) => {
           const cardLocale = cardLocales[index];
           const shouldShow = (selectedLocale === 'ALL' || cardLocale === selectedLocale);
           
           if (shouldShow) {
             card.style.setProperty('display', 'flex', 'important');
             card.style.setProperty('visibility', 'visible', 'important');
             visibleCount++;
             console.log('Showing card', index, '(', cardLocale, ')');
           } else {
             card.style.setProperty('display', 'none', 'important');
             card.style.setProperty('visibility', 'hidden', 'important');
             console.log('Hiding card', index, '(', cardLocale, ')');
           }
         });
         
         console.log('Total visible cards:', visibleCount);
         
         // Remove existing "no results" message
         const existingMsg = document.querySelector('.no-results-message');
         if (existingMsg) {
           existingMsg.remove();
         }
         
         // Show "no results" message if needed
         if (visibleCount === 0) {
           const noResultsMsg = document.createElement('div');
           noResultsMsg.className = 'no-results-message';
           noResultsMsg.style.cssText = 'padding: 20px; text-align: center; color: #666; font-style: italic; margin-top: 20px;';
           noResultsMsg.textContent = 'No articles found for the selected language.';
           articlesGrid.parentElement.appendChild(noResultsMsg);
         }
       }
       
       // Add change event listener
       filter.addEventListener('change', function(e) {
         const selectedLocale = e.target.value.toUpperCase(); // Convert to uppercase for comparison
         applyFilter(selectedLocale);
       });
       
       // Apply initial filter on page load (English by default)
       const initialLocale = filter.value.toUpperCase();
       applyFilter(initialLocale);
       
       console.log('Filter initialized successfully!');
     }
     
     // Initialize when DOM is ready
     if (document.readyState === 'loading') {
       document.addEventListener('DOMContentLoaded', initFilter);
     } else {
       initFilter();
     }
   })();
   </script>

.. important::

   AWS and Neuron provide links to external articles and posts to help you discover them, but do not commission or own any content not created by AWS employees. This list is curated based on internal and customer recommendations. 

**Want to add your article?** Go to `https://github.com/aws-neuron/aws-neuron-sdk <https://github.com/aws-neuron/aws-neuron-sdk>`_, edit ``about-neuron/news-and-blogs/news-and-blogs.yaml`` to add your submission, and submit a pull request. 



