//TODO: aquire all visible comment threads
//Order by replies
//Order by upvotes for comment threads within an order of magnitude to each other
//expand all comments under comment threads sequentially in the order defined above. Do this by $$ing within the OP comment of the comment thread
//send all to md file. If in performance mode send only the top 10 comment threads.
//use to create prompt
//send prompt to LLM
//ask for general summary, ask for for and against analysis, ask for summary of reasons for and reasons against
//screen user so that the prompt is a FOR AND AGAINST question

const { connect } = require('puppeteer-real-browser');
const fs = require('fs')

let remaining_rqs = 100000

async function wait_for_morescroll(page, cur_height, timeout = 5000, interval = 500) {
    for (i = 0; i < Math.ceil(timeout/interval); i++) {
        await new Promise(r => setTimeout(r, interval));
        let new_height = await page.evaluate(() => document.body.scrollHeight);
        if (cur_height != new_height) {
            return new_height;
        }
        else {
            continue;
        }
    }
    return cur_height;
}

async function bottomout_scroll(page, start_height, timeout = 5000, interval = 500) {
    let cur_height = start_height;
    while (true) {
        await page.evaluate(() => {
            window.scrollTo(0, document.body.scrollHeight);
        });
        new_height = await wait_for_morescroll(page, cur_height, timeout, interval);
        if (cur_height == new_height) {
            break;
        }
        cur_height = new_height;
    }
    return cur_height;
}

async function is_clickable(elem) {
    let box = await elem.boundingBox();
    if (box != null) {
        return true;
    }
    return false;
}

async function find_clickables(handle, selector) {
    let unfiltered = await handle.$$(selector);
    let filtered = [];
    for (let i = 0; i < unfiltered.length; i++) {
        if (await is_clickable(unfiltered[i])) {
            filtered.push(unfiltered[i]);
        }
    }
    return filtered;
}

async function childupdate_wait(root_el, child_comment_selector, timeout = 2500, interval = 500, onlymore=false, onlyless=false) {
    //console.log(await root_el.evaluate(r => r.tagName))
    let org_childnum = await root_el.$$(child_comment_selector);
    for (let i = 0; i < Math.ceil(timeout/interval); i++) {
        await new Promise(r => setTimeout(r, interval));
        let cur_childnum = await root_el.$$(child_comment_selector);
        //console.log(cur_childnum)
        if (onlymore && org_childnum.length < cur_childnum.length) {
            return true;
        }
        else if (onlyless && org_childnum.length > cur_childnum.length) {
            return true;
        }
        else if (org_childnum.length != cur_childnum.length) {
            return true;
        }
        else {
            continue;
        }
    }
    return false;
}

async function unique_comm_select(el) {
    return `shreddit-comment[permalink="${await el.evaluate(comment => comment.getAttribute("permalink"))}"`;
}

async function traverse_comments(cur_selector, reply_selector, newpage_selector, child_selector, root_selector, content_selector, rq_ct, cur_rqs, browser, page) {
    let rqs_used = cur_rqs;
    let comment_chain_ended = true;
    let cur_el = await page.waitForSelector(cur_selector);
    let cur_el_content = "COMMENT COULD NOT BE RETRIEVED"
    try {
        cur_el_content = await (await cur_el.$(content_selector)).evaluate(content => content.innerText);
    }
    catch {

    }
    let discussion_up_to_cur = []

    if (rqs_used < rq_ct) {
        //Expand comment thread if needed
        let more_reply_button = await find_clickables(cur_el, reply_selector);
        while (more_reply_button.length > 0 && rqs_used < rq_ct) {
            try {
                await more_reply_button[0].click();
                await Promise.any([
                    page.waitForNetworkIdle(),
                    childupdate_wait(cur_el, child_selector)
                ]);
                await more_reply_button[0].dispose();
                rqs_used += 1;
            }
            catch {
                
            }
            more_reply_button = await find_clickables(cur_el, reply_selector);
        }
        //Go to new page to view test of comment thread if needed
        let newpage_button = await find_clickables(cur_el, newpage_selector);
        if (newpage_button.length > 0) {
            comment_chain_ended = false;
            let link = await newpage_button[0].evaluate(hyperlink => hyperlink.href);
            const new_page = await browser.newPage();
            await new_page.goto(link);
            rqs_used += 1;
            discussion_up_to_cur = await traverse_comments(root_selector, reply_selector, newpage_selector, child_selector, root_selector, content_selector, rq_ct, rqs_used, browser, new_page);
            await new_page.close();
        }
        //Traverse to child comments
        cur_el = await page.waitForSelector(cur_selector);
        let child_comments = await cur_el.$$(child_selector);
        if (child_comments.length > 0) { 
            comment_chain_ended = false; 
        }
        for (let i = 0; i < child_comments.length; i++) {
            let unique_selector = await unique_comm_select(child_comments[i]);
            let child_discussions = await traverse_comments(unique_selector, reply_selector, newpage_selector, child_selector, root_selector, content_selector, rq_ct, rqs_used, browser, page);
            for (let j = 0; j < child_discussions.length; j++) {
                child_discussions[j].push(cur_el_content);
            }
            discussion_up_to_cur = discussion_up_to_cur.concat(child_discussions);
        }
    }

    if (comment_chain_ended) {
        discussion_up_to_cur = [[cur_el_content]];
    }
    return discussion_up_to_cur;
}

function calculate_value(replies, upvotes) {
    return replies * (upvotes / 10)
}

function exportToCSV(filename, data) {
    let formatted_data = data.map(row => row.map(cell => `"${cell}"`).join(",")).join("\n");
    fs.writeFileSync(filename, formatted_data);  // Write to file
    console.log(`${filename} has been saved!`);
  }

async function start() {
    const { browser, page } = await connect({

        headless: false,

        args: [],

        customConfig: {},

        turnstile: true,

        connectOption: {},

        disableXvfb: false,

        ignoreAllFlags: false

        // proxy:{
        //     host:'<proxy-host>',
        //     port:'<proxy-port>',
        //     username:'<proxy-username>',
        //     password:'<proxy-password>'
        // }
    });

    await page.goto('https://www.reddit.com/r/socialskills/comments/1en5k0t/what_can_i_respond_with_instead_of_always_saying/', { waitUntil: "domcontentloaded" });

    //reveal all root comments
    let cur_height = 0
    while (true) {
        cur_height = await bottomout_scroll(page, cur_height, 1000, 500);
        try {
            morecomment_button = await page.$('span ::-p-text(View more comments)');
            await morecomment_button.click();
            await morecomment_button.dispose();
            await wait_for_morescroll(page, cur_height);
        }
        catch {
            break;
        }
        // await page
        // .locator('span')
        // .filter(button => button.innerText === 'View more comments')
        // .click();
    }

    //Get the root comment of all comment threads
    let rootcomms = await page.$$('shreddit-comment[depth="0"]');

    //score root comments and filter
    let scored_threads = []
    for (let i = 0; i < rootcomms.length; i++) {
        if (!(await rootcomms[i].evaluate(e => e.hasAttribute("is-comment-deleted")))) {
            let permalink = await rootcomms[i].evaluate(comment => comment.getAttribute("permalink"))
            let cur_thread = {selector: `shreddit-comment[permalink="${permalink}"]`, link: permalink, score: calculate_value(1, await rootcomms[i].evaluate(comment => comment.getAttribute("score")))};
            scored_threads.push(cur_thread);
            rootcomms[i].dispose();
        }
    }

    //sort comment threads by a score assigned to their root comment
    scored_threads.sort((com1, com2) => (com1.score - com2.score)).reverse();
    // let towrite = ""
    // for (let i = 0; i < scored_threads.length; i++) {
    //     let comment_obj = `Score: ${scored_threads[i].score}, Link: ${scored_threads[i].link}`;
    //     towrite += comment_obj + "\n";
    // }
    // fs.writeFileSync('prompt.md', towrite, (err) => {if (err) throw err;});

    let more_comments_selector = '::-p-xpath(./*[not(self::shreddit-comment) and not(self::a)]//button[./*[text()=" more replies" or text()=" more reply"]])';
    let new_page_selector = '::-p-xpath(./a[./*[text()="More replies" or text()=" more replies" or text()=" more reply"]])';
    let child_comment_selector = '::-p-xpath(./shreddit-comment)';
    let root_selector = '::-p-xpath(//shreddit-comment)';
    let comment_content_selector = '::-p-xpath(./*[not(self::shreddit-comment) and @slot="comment"])'

    let comment_response = [['initial_comment', 'responding_comment']]
    for (let i = 0; i < scored_threads.length; i++) {
        let cur_element = await page.$(`${scored_threads[i].selector}`);
        if (await cur_element.evaluate(e => e.hasAttribute("collapsed"))) {
            await cur_element.evaluate(e => e.removeAttribute("collapsed"));
        }
        let discussions = await traverse_comments(scored_threads[i].selector, more_comments_selector, new_page_selector, child_comment_selector, root_selector, comment_content_selector, remaining_rqs, 0, browser, page);
        //console.log(`thread: ${await cur_element.evaluate(comment => comment.getAttribute("author"))}, reply buttons: ${remaining_rqs}`);
        if (remaining_rqs <= 0) {
            break;
        }

        for (let i1 = 0; i1 < discussions.length; i1++) {
            let discussion = discussions[i1];
            if (discussion.length > 1) {
                for (let j = discussion.length - 1; j > 0; j--) {
                    comment = discussion[j]
                    comment_response.push([discussion[j], discussion[j-1]])
                }
            }
            // console.log("--------------------------------------------------------------------");
        }
    }
    exportToCSV('output.csv', comment_response)

    // let root_test = 'shreddit-comment[permalink="/r/socialskills/comments/1en5k0t/comment/lh3tdw9/"]'
    // let more_comments_selector = '::-p-xpath(./*[not(self::shreddit-comment) and not(self::a)]//button[./*[text()=" more replies" or text()=" more reply"]])';
    // let new_page_selector = '::-p-xpath(./a[./*[text()="More replies" or text()=" more replies" or text()=" more reply"]])';
    // let child_comment_selector = '::-p-xpath(./shreddit-comment)';
    // let root_selector = '::-p-xpath(//shreddit-comment)';
    // let comment_content_selector = '::-p-xpath(./*[not(self::shreddit-comment) and @slot="comment"])'
    // let discussions = await traverse_comments(root_test, more_comments_selector, new_page_selector, child_comment_selector, root_selector, comment_content_selector, remaining_rqs, 0, browser, page);
    // for (let i = 0; i < discussions.length; i++) {
    //     for (let j = discussions[i].length - 1; j >= 0; j--) {
    //         console.log(discussions[i][j]);
    //         console.log()
    //     }
    //     console.log("--------------------------------------------------------------------")
    // }

    // let root_test = await page.$('shreddit-comment[author="ObviousMousse4768"]')
    // let test = await root_test.$$('::-p-xpath(./shreddit-comment)');
    // console.log(test.length);
    // for (let i = 0; i < test.length; i++) {
    //     console.log(await test[i].evaluate(comment => comment.getAttribute("author")));
    // }

    //console.log(remaining_rqs);

    await browser.close();
}

start();
