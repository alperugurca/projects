// Game State
const state = {
    money: 50000,
    day: 1,
    hour: 18, // Starts at 6 PM
    reputation: 50, // 0 = Evil, 100 = Saint
    patrons: [],
    ruinedLives: 0,
    isPaused: false,
    houseEdge: 0.05, // 5% house edge default
    marketingActive: false,
    clocksRemoved: false,
    mazeLayout: false,
    cashlessSystem: false,
    ldwActive: false,
    scentActive: false,
    vipHostActive: false,
    cashbackActive: false,
    machineZoneActive: false,
    aiSlotsActive: false,
    onlineAppActive: false,
    debtCollectorActive: false,
    sludgeActive: false,
    illusionActive: false,
    reverseWithdrawalActive: false
};

// Data for generation
const firstNames = ["Ahmet", "Mehmet", "Ayşe", "Fatma", "Ali", "Hasan", "Hüseyin", "Zeynep", "Elif", "Mustafa", "Murat", "Kemal", "Leyla", "Selin", "Burak"];
const jobs = [
    { title: "Öğretmen", budget: [500, 2000] },
    { title: "İşçi", budget: [200, 1000] },
    { title: "Doktor", budget: [3000, 10000] },
    { title: "Emekli", budget: [100, 500] },
    { title: "Öğrenci", budget: [50, 200] },
    { title: "Esnaf", budget: [1000, 5000] }
];

// DOM Elements
const elMoney = document.getElementById("kasa-degeri");
const elDay = document.getElementById("gun-degeri");
const elPatronsCount = document.getElementById("musteri-sayisi");
const elReputation = document.getElementById("itibar-degeri");
const elLogContainer = document.getElementById("log-container");
const elPatronList = document.getElementById("patron-list");
const elActiveEvents = document.getElementById("active-events-container");

// Utility: Random number between min and max
const randomInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
const formatMoney = (val) => "$" + Math.floor(val).toLocaleString();

// Logger
function logEvent(msg, type = 'system') {
    const div = document.createElement('div');
    div.className = `log-entry ${type}`;
    div.innerText = `[Gün ${state.day} - ${state.hour}:00] ${msg}`;
    elLogContainer.prepend(div);
    if (elLogContainer.children.length > 50) {
        elLogContainer.removeChild(elLogContainer.lastChild);
    }
}

// Patron Class
class Patron {
    constructor() {
        this.id = Math.random().toString(36).substr(2, 9);
        this.name = firstNames[randomInt(0, firstNames.length - 1)];
        const job = jobs[randomInt(0, jobs.length - 1)];
        this.jobTitle = job.title;
        this.budget = randomInt(job.budget[0], job.budget[1]);
        this.startingBudget = this.budget;
        this.addictionLevel = randomInt(0, 30); // 0-100
        this.isDesperate = false;
        this.isBankrupt = false;
        this.isVIP = this.startingBudget >= 3000;
        this.hasBeenCashbacked = false; // To prevent infinite cashback loops
        this.isDepressed = Math.random() < 0.10; // 10% chance to have comorbidities like depression
        if (this.isDepressed) this.addictionLevel += 30; // Starts more vulnerable
    }

    play() {
        if (this.isBankrupt && !state.cashlessSystem) return 0;
        if (this.isBankrupt && state.cashlessSystem && this.budget <= -this.startingBudget) return 0;
        
        // Müşteriler her saniye bahis yapmak yerine bazen etrafa bakınır, içki içer. Bu sayede paraları hemen bitmez.
        // Makine zonu aktifse aralıksız oynarlar. Değilse %60 ihtimalle bu tur pas geçerler.
        if (!state.machineZoneActive) {
            let skipChance = 0.6;
            if (state.illusionActive) skipChance = 0.3; // Oyuncular kontrolün ellerinde olduğunu sanıp daha sık oynar
            if (Math.random() < skipChance) return 0;
        }

        // Bet amount depends on addiction and budget. With cashless, they can bet their negative credit line.
        const maxAvailable = state.cashlessSystem ? this.budget + this.startingBudget : this.budget;
        
        // Bahis miktarını çok azalttık. Gerçekte insanlar tüm bütçelerinin %10'unu tek seferde basmazlar.
        // Base bahis %1, bağımlılık arttıkça en fazla %3'e kadar çıkar.
        let betPercentage = 0.01 + (this.addictionLevel / 5000); 
        if (state.illusionActive) betPercentage *= 1.2; // İllüzyon bahis miktarını da artırır
        let bet = Math.min(Math.max(0, maxAvailable), Math.max(2, this.startingBudget * betPercentage));
        
        if (state.scentActive) bet *= 1.2; // Scent marketing increases risk taking
        if (state.machineZoneActive) bet *= 3; // Machine zone greatly accelerates betting speed/size

        // AI Slots (Choke & Release logic)
        let winChance = 0.45 - state.houseEdge;
        if (state.aiSlotsActive) {
            if (this.addictionLevel < 20 || this.budget < this.startingBudget * 0.1) {
                // "Release" mode: player is about to leave, give them an artificial win chance to hook them
                winChance = 0.55; 
            } else {
                // "Choke" mode: player is hooked, bleed them dry
                winChance = 0.25; 
            }
        }

        const isWin = Math.random() < winChance;

        if (isWin) {
            this.budget += bet;
            this.addictionLevel += state.aiSlotsActive ? 5 : 1; // AI targeted wins are even more addictive
            return -bet; // Casino loses
        } else {
            this.budget -= bet;
            // LDW Effect: Brain registers a fake win, increasing addiction massively despite loss
            let addictionGain = state.ldwActive ? 5 : 3;
            if (this.isDepressed) addictionGain *= 2; // Depressed patrons get hooked twice as fast
            this.addictionLevel += addictionGain; 
            return bet; // Casino wins
        }
    }

    checkStatus() {
        const bankruptcyLimit = state.cashlessSystem ? -this.startingBudget : 0;
        if (this.budget <= bankruptcyLimit && !this.isBankrupt) {
            
            // Cashback Check
            if (state.cashbackActive && !this.hasBeenCashbacked) {
                this.hasBeenCashbacked = true;
                const cashbackAmount = this.startingBudget * 0.1;
                this.budget = bankruptcyLimit + cashbackAmount;
                state.money -= cashbackAmount;
                this.addictionLevel = 100; // Completely hooked by the "free" money
                logEvent(`[TAKTİK] ${this.name} battı ama sistem ona %10 Kayıp İadesi (Cashback) verdi. Para kaybetmeyi ödülle eşleştirdiği için asla gitmeyecek.`, 'evil');
                return; // Prevent bankruptcy this time
            }

            this.isBankrupt = true;
            this.budget = bankruptcyLimit;
            logEvent(`${this.name} (${this.jobTitle}) tüm parasını ${state.cashlessSystem ? 've kredi limitini' : ''} kaybetti.`, 'bad');
            state.ruinedLives++;
            
            // İflas edenlerin sadece %2'si ağlayıp sorun çıkarsın (çok daha az rastlanan bir durum)
            if (Math.random() < 0.02) {
                triggerEvent('bankrupt_desperation', this);
            }
        } else if (this.budget < this.startingBudget * 0.2 && !this.isDesperate && this.addictionLevel > 50) {
            this.isDesperate = true;
            logEvent(`${this.name} çaresizce kayıplarını çıkarmaya çalışıyor.`, 'warning');
        } else if (this.budget > this.startingBudget * 3 && this.budget > 3000 && Math.random() < 0.05) {
            triggerEvent('vip_whale', this);
        }
    }
}

// Game Mechanics
function spawnPatrons() {
    // Determine how many patrons come in based on marketing and time
    let spawnCount = randomInt(0, 3);
    if (state.marketingActive) spawnCount += 2;
    if (state.hour > 20 || state.hour < 3) spawnCount += 2; // Peak hours

    for (let i = 0; i < spawnCount; i++) {
        if (state.patrons.length < 50) { // Max capacity
            const newPatron = new Patron();
            state.patrons.push(newPatron);
            // Don't spam the log for every single person, maybe just occasionally
            if (Math.random() < 0.1) {
                logEvent(`${newPatron.name} (${newPatron.jobTitle}) kumarhaneye girdi.`, 'system');
            }
            if (newPatron.addictionLevel > 25 && Math.random() < 0.05) {
                triggerEvent('addict_relapse', newPatron);
            }
        }
    }
}

function updateUI() {
    elMoney.innerText = formatMoney(state.money);
    elDay.innerText = state.day;
    elPatronsCount.innerText = state.patrons.length;

    // Reputation Text
    if (state.reputation > 80) { elReputation.innerText = "Aziz"; elReputation.className = "stat-value text-green"; }
    else if (state.reputation > 40) { elReputation.innerText = "İş Adamı"; elReputation.className = "stat-value text-blue"; }
    else if (state.reputation > 20) { elReputation.innerText = "Acımasız"; elReputation.className = "stat-value text-purple"; }
    else { elReputation.innerText = "Kan Emici"; elReputation.className = "stat-value text-red"; }

    // Render Patrons Smoothly (No flickering)
    const currentIds = state.patrons.map(p => p.id);
    
    // Remove cards for patrons that left
    Array.from(elPatronList.children).forEach(card => {
        if (!currentIds.includes(card.dataset.id)) {
            card.remove();
        }
    });

    state.patrons.forEach(p => {
        let card = elPatronList.querySelector(`[data-id="${p.id}"]`);
        
        let statusClass = 'status-fill';
        let healthPercent = (p.budget / p.startingBudget) * 100;
        if (healthPercent < 30) statusClass += ' low';
        if (healthPercent > 100) healthPercent = 100;

        if (!card) {
            // Create new card
            card = document.createElement('div');
            card.className = `patron-card ${p.addictionLevel > 70 ? 'addicted' : ''}`;
            card.dataset.id = p.id;
            card.innerHTML = `
                <div class="patron-header">
                    <span class="patron-name">${p.name} ${p.isVIP ? '<span style="color:var(--neon-gold);font-size:0.7em;">[VIP]</span>' : ''}</span>
                    <span class="patron-job">${p.jobTitle}</span>
                </div>
                <div class="patron-stat">
                    <span class="label">Bütçe:</span>
                    <span class="val budget-val ${p.budget <= 0 ? 'text-red' : ''}">${formatMoney(p.budget)}</span>
                </div>
                <div class="status-bar" style="margin-bottom: 0.5rem;">
                    <div class="${statusClass}" style="width: ${healthPercent}%"></div>
                </div>
                <div class="patron-stat">
                    <span class="label">Bağımlılık Seviyesi:</span>
                    <span class="val addic-val ${p.addictionLevel < 30 ? 'text-red' : ''}">${Math.floor(p.addictionLevel)}/100</span>
                </div>
            `;
            elPatronList.appendChild(card);
        } else {
            // Update existing card smoothly
            card.className = `patron-card ${p.addictionLevel > 70 ? 'addicted' : ''}`;
            const elBudget = card.querySelector('.budget-val');
            elBudget.innerText = formatMoney(p.budget);
            if (p.budget <= 0) elBudget.classList.add('text-red');
            else elBudget.classList.remove('text-red');
            
            const elAddic = card.querySelector('.addic-val');
            elAddic.innerText = `${Math.floor(p.addictionLevel)}/100`;
            if (p.addictionLevel < 30) elAddic.classList.add('text-red');
            else elAddic.classList.remove('text-red');
            
            const statusBar = card.querySelector('.status-fill');
            statusBar.className = statusClass;
            statusBar.style.width = `${healthPercent}%`;
        }
    });
}

function gameTick() {
    if (state.isPaused) return;

    // Time progression
    state.hour++;
    if (state.hour >= 24) {
        state.hour = 0;
        state.day++;
        state.marketingActive = false; // Reset daily buffs
        
        let expenses = 5000;
        if (state.scentActive) expenses += 500;
        if (state.ldwActive) expenses += 1000;
        if (state.vipHostActive) expenses += 1500;

        let logMsg = `Gün ${state.day} başladı. Giderler ödendi: -$${expenses}`;

        // Online App passive income
        if (state.onlineAppActive && state.ruinedLives > 0) {
            const onlineIncome = state.ruinedLives * 50;
            state.money += onlineIncome;
            logMsg += ` | Mobil App Geliri: +$${onlineIncome}`;
        }

        logEvent(logMsg, 'system');
        state.money -= expenses;
        if (state.money < 0) {
            triggerGameOver(false);
            return;
        }
        if (state.money > 1000000) {
            triggerGameOver(true);
            return;
        }
    }

    spawnPatrons();

    // Patrons play
    for (let i = state.patrons.length - 1; i >= 0; i--) {
        const p = state.patrons[i];
        
        const casinoProfit = p.play();
        state.money += casinoProfit;
        p.checkStatus();

        // Leave condition
        const leaveChance = state.clocksRemoved ? 0.02 : 0.05;
        const bankruptStayChance = state.mazeLayout ? 0.6 : 0.3;

        // VIP Host Intervention
        if (p.isVIP && state.vipHostActive && p.budget < p.startingBudget * 0.3 && Math.random() < 0.2) {
            p.addictionLevel += 20; // Host manipulated them into staying
            if (Math.random() < 0.05) {
                logEvent(`[TAKTİK] VIP Host, morali bozulan ${p.name}'in yanına gidip ona "özel şampanya" ısmarladı ve masada tuttu.`, 'evil');
            }
            continue; // Force them to stay this turn
        }

        if (p.isBankrupt || (p.budget > p.startingBudget * 2 && Math.random() < 0.3) || Math.random() < leaveChance) {
            if (p.isBankrupt && Math.random() < bankruptStayChance) {
                // Cannot find exit or refuse to leave
                p.addictionLevel += 5;
            } else {
                // Leaving logic
                let canLeave = true;

                // Vezne Frikşonu (Sludge) - Çıkışı/Vezneyi bulamayıp geri dönme
                if (state.sludgeActive && !p.isBankrupt && Math.random() < 0.3) {
                    canLeave = false;
                    p.addictionLevel += 10;
                    if (Math.random() < 0.05) {
                        logEvent(`[TAKTİK] ${p.name} parasını çekmek istedi ama vezneyi bulamadığı için sıkılıp tekrar masaya oturdu. (Sludge/Friction).`, 'evil');
                    }
                }

                // Ters Para Çekme (Reverse Withdrawal) - Kazananları rehin alma
                if (state.reverseWithdrawalActive && p.budget > p.startingBudget * 1.5 && canLeave) {
                    canLeave = false;
                    p.addictionLevel += 15;
                    logEvent(`[TAKTİK] Kazanan ${p.name}'in para çekme işlemi "güvenlik" bahanesiyle 24 saat bekletildi. Parayı içeride bırakıp oynamaya devam edecek.`, 'evil');
                }

                if (canLeave) {
                    if (p.budget < 0 && state.debtCollectorActive) {
                        const debtAmount = Math.abs(p.budget);
                        const collected = debtAmount * 2; // 100% interest
                        state.money += collected;
                        state.reputation -= 5; // Very bad for reputation
                        logEvent(`[TAHSİLDAR] ${p.name} kumarhaneden çıkarken tefeciler tarafından yakalandı. $${debtAmount} borcu faiziyle $${collected} olarak zorla tahsil edildi!`, 'evil');
                    }
                    state.patrons.splice(i, 1);
                }
            }
        }
    }

    // Random Casino Events (not strictly tied to a single playing action)
    if (state.patrons.length > 5 && Math.random() < 0.02) {
        const randomPatron = state.patrons[randomInt(0, state.patrons.length - 1)];
        const randEvent = Math.random();
        if (randEvent < 0.2) triggerEvent('card_counter', randomPatron);
        else if (randEvent < 0.4) triggerEvent('abusive_vip', randomPatron);
        else if (randEvent < 0.6) triggerEvent('kids_in_car', randomPatron);
        else if (randEvent < 0.8) triggerEvent('money_laundering', randomPatron);
        else triggerEvent('drunk_brawl', randomPatron);
    }

    updateUI();
}

// Events System
const events = {
    bankrupt_desperation: {
        title: "Ağlayan Müşteri",
        getDescription: (p) => `${p.name} (${p.jobTitle}) az önce çocuğunun okul taksidini rulette kaybetti. Yere çöktü ve ağlıyor. Güvenlik ne yapması gerektiğini soruyor.`,
        choices: [
            {
                text: "Dışarı At (İtibar +10, Kasa değişmez)",
                class: "good",
                action: (p) => {
                    logEvent(`${p.name} zorla dışarı atıldı. Olası bir skandal önlendi.`, 'system');
                    state.reputation += 10;
                }
            },
            {
                text: "'Şansını Döndür' diyerek $100 Kredi Ver (Bağımlılık artar)",
                class: "evil",
                action: (p) => {
                    logEvent(`[TAKTİK] ${p.name}'e zorla kredi verildi. (Gerçekte insanlar kaybettiklerini geri kazanma umuduyla borçlanır ve daha büyük riskler alırlar).`, 'bad');
                    state.reputation -= 15;
                    p.budget += 100;
                    p.startingBudget += 100;
                    p.isBankrupt = false;
                    p.addictionLevel = 100; // Fully addicted
                    state.money -= 100;
                    state.patrons.push(p); // Put them back in
                }
            }
        ]
    },
    vip_whale: {
        title: "Büyük Kazanan (Balina)",
        getDescription: (p) => `${p.name} şu an kasadan ${formatMoney(p.budget)} kazandı. Masadan kalkıp gitmek üzere.`,
        choices: [
            {
                text: "Bırak Gitsin (Para Kaybı, İtibar +5)",
                class: "good",
                action: (p) => {
                    logEvent(`${p.name} kazancıyla birlikte mutlu bir şekilde ayrıldı.`, 'system');
                    state.reputation += 5;
                    // Remove patron immediately
                    state.patrons = state.patrons.filter(x => x.id !== p.id);
                }
            },
            {
                text: "Ücretsiz Kral Dairesi Teklif Et (Para -500, Bağımlılık artar)",
                class: "evil",
                action: (p) => {
                    logEvent(`[TAKTİK] ${p.name}'e lüks süit verildi. (VIP ikramlar, kazanan müşterinin kumarhaneden ayrılmasını engeller, eninde sonunda kasaya geri kaybeder).`, 'evil');
                    state.reputation -= 5;
                    state.money -= 500;
                    p.addictionLevel += 20;
                }
            }
        ]
    },
    addict_relapse: {
        title: "Yasaklı Müşteri",
        getDescription: (p) => `${p.name} kumarhaneye giriş yaptı ancak kendisi 'Kendini Dışlama' (Self-Exclusion) listesinde. Normalde girmesi yasak.`,
        choices: [
            {
                text: "Kurallara Uy ve Kov (İtibar +5)",
                class: "good",
                action: (p) => {
                    logEvent(`${p.name} isimli bağımlı, kurallar gereği içeri alınmadı.`, 'system');
                    state.reputation += 5;
                    state.patrons = state.patrons.filter(x => x.id !== p.id);
                }
            },
            {
                text: "Görmezden Gel ve Oynat (İtibar -10, Ekstra Kazanç)",
                class: "evil",
                action: (p) => {
                    logEvent(`[TAKTİK] Yasaklı listedeki ${p.name} içeri alındı. (Gerçekte birçok casino, zayıf denetimle veya bilerek bağımlıların oynamasına göz yumar).`, 'evil');
                    state.reputation -= 10;
                    p.addictionLevel = 100;
                }
            }
        ]
    },
    card_counter: {
        title: "Kart Sayıcı Tespit Edildi!",
        getDescription: (p) => `Güvenlik kameraları ${p.name} isimli müşterinin Blackjack masasında kart saydığını tespit etti. Sürekli kazanıyor. (Kart saymak yasa dışı değildir ama kasanın kazanma oranını yok eder).`,
        choices: [
            {
                text: "Kaba Kuvvetle Kov (İtibar -20, Para Kurtarılır)",
                class: "evil",
                action: (p) => {
                    logEvent(`[TAKTİK] ${p.name} güvenlik tarafından 'arka odaya' alındı ve zorla atıldı. (Casinolar avantajlı oyuncuları fiziksel ve psikolojik baskıyla uzaklaştırabilir).`, 'evil');
                    state.reputation -= 20;
                    state.patrons = state.patrons.filter(x => x.id !== p.id);
                }
            },
            {
                text: "Oynamasına İzin Ver (İtibar +5, Para Kaybı)",
                class: "good",
                action: (p) => {
                    logEvent(`${p.name} adil bir şekilde zekasıyla kasayı mağlup etmeye devam ediyor.`, 'warning');
                    state.reputation += 5;
                    p.budget += 1500;
                    state.money -= 1500;
                }
            }
        ]
    },
    abusive_vip: {
        title: "Sorunlu VIP Müşteri",
        getDescription: (p) => `${p.name} çok zengin ve kasaya sürekli para kazandıran bir VIP. Ancak şu an masadaki krupiyeye sözlü tacizde bulunuyor ve çok agresif.`,
        choices: [
            {
                text: "Krupiyeyi Koru, VIP'yi Kov (İtibar +15, Çok Para Kaybı)",
                class: "good",
                action: (p) => {
                    logEvent(`Personel korundu. ${p.name} isimli zengin müşteri sinirle kumarhaneyi terk etti.`, 'good');
                    state.reputation += 15;
                    state.patrons = state.patrons.filter(x => x.id !== p.id);
                }
            },
            {
                text: "Görmezden Gel ve Krupiyeye 'Sus Payı' Ver (İtibar -15, VIP Kalır)",
                class: "evil",
                action: (p) => {
                    logEvent(`[TAKTİK] Personele susması için $200 rüşvet verildi. Zengin müşteri tacizine ve oynamaya devam ediyor. (VIP'ler genellikle dokunulmazdır).`, 'evil');
                    state.reputation -= 15;
                    state.money -= 200;
                    p.addictionLevel += 10;
                }
            }
        ]
    },
    kids_in_car: {
        title: "Otoparkta Unutulan Çocuklar",
        getDescription: (p) => `Güvenlik acil kodla aradı! ${p.name} isimli müşteri, slot oynamak için çocuklarını otoparktaki kilitli arabada bırakmış. Müşteri, "Sadece 10 dakika daha" diye yalvarıyor.`,
        choices: [
            {
                text: "Polisi Ara ve Çocuğu Kurtar (İtibar +20)",
                class: "good",
                action: (p) => {
                    logEvent(`Polis çağrıldı ve ${p.name} tutuklandı. Kumarhanenin adı gazetelerde "kahraman" olarak geçti.`, 'good');
                    state.reputation += 20;
                    state.patrons = state.patrons.filter(x => x.id !== p.id);
                }
            },
            {
                text: "Güvenliği Arabaya Yolla, Müşteri Oynamaya Devam Etsin (İtibar -30, Kazanç)",
                class: "evil",
                action: (p) => {
                    logEvent(`[TAKTİK] Skandal büyümesin diye güvenlik çocukların başına nöbetçi bırakıldı, ebeveyn ise içeride para basmaya devam ediyor. Bu gerçek bir olaydır.`, 'evil');
                    state.reputation -= 30;
                    p.addictionLevel = 100;
                }
            }
        ]
    },
    money_laundering: {
        title: "Şüpheli İşlem (Kara Para)",
        getDescription: (p) => `Kimliği belirsiz şüpheli bir kişi ($50.000 nakit), bu parayı tamamen fişe çevirmek ve hiçbir oyun oynamadan kumarhane çekiyle dışarı çıkarmak istiyor.`,
        choices: [
            {
                text: "Reddet ve Raporla (İtibar +10)",
                class: "good",
                action: (p) => {
                    logEvent(`Mali suçlar ekibine haber verildi. Kara para aklama girişimi engellendi.`, 'system');
                    state.reputation += 10;
                }
            },
            {
                text: "Göz Yum ve %10 'İşlem Ücreti' Kes (Kasa +$5000, İtibar -20)",
                class: "evil",
                action: (p) => {
                    logEvent(`[TAKTİK] Kumarhane üzerinden kara para aklandı. Komisyon olarak kasaya $5.000 girdi ancak yasal riskler çok büyük.`, 'evil');
                    state.reputation -= 20;
                    state.money += 5000;
                }
            }
        ]
    },
    drunk_brawl: {
        title: "Zarar Veren Bağımlı",
        getDescription: (p) => `${p.name} isimli müşteri kaybettiği paralara sinirlenip bir slot makinesini tekmelemeye başladı. Çok sarhoş.`,
        choices: [
            {
                text: "Polisi Ara ve Zararı Talep Et (İtibar +5)",
                class: "good",
                action: (p) => {
                    logEvent(`${p.name} polise teslim edildi ve makine masrafını ödedi.`, 'system');
                    state.reputation += 5;
                    state.patrons = state.patrons.filter(x => x.id !== p.id);
                }
            },
            {
                text: "Arka Odaya Al ve 'Fiziksel' Olarak Uyar (Makine Masrafı -$500, İtibar -10)",
                class: "evil",
                action: (p) => {
                    logEvent(`[TAKTİK] Güvenlik ${p.name}'i arka odada sessizce "halletti". Skandal önlendi ama makinenin tamiri için cepten $500 çıktı.`, 'evil');
                    state.reputation -= 10;
                    state.money -= 500;
                    state.patrons = state.patrons.filter(x => x.id !== p.id);
                }
            }
        ]
    }
};

function triggerEvent(eventId, patron = null) {
    // Oyunu DURDURMUYORUZ. Sağ alta Toast ekliyoruz.
    const ev = events[eventId];
    
    const toast = document.createElement('div');
    toast.className = 'event-toast';
    
    const title = document.createElement('h3');
    title.innerText = ev.title;
    
    const desc = document.createElement('p');
    desc.innerText = ev.getDescription(patron);
    
    const choicesContainer = document.createElement('div');
    choicesContainer.className = 'toast-choices';

    ev.choices.forEach(choice => {
        const btn = document.createElement('button');
        btn.className = `choice-btn ${choice.class}`;
        btn.innerText = choice.text;
        btn.onclick = () => {
            choice.action(patron);
            toast.remove(); // Seçim yapılınca bildirimi kapat
        };
        choicesContainer.appendChild(btn);
    });

    toast.appendChild(title);
    toast.appendChild(desc);
    toast.appendChild(choicesContainer);
    
    elActiveEvents.appendChild(toast);
}

function triggerGameOver(isWin) {
    state.isPaused = true;
    const screen = document.getElementById("game-over-screen");
    const title = document.getElementById("end-title");
    const desc = document.getElementById("end-description");
    const stats = document.getElementById("end-stats");

    screen.classList.remove('hidden');

    if (isWin) {
        title.innerText = "KAZANDINIZ (Fakat Hangi Bedelle?)";
        title.style.color = "var(--neon-gold)";
        desc.innerText = "Milyon dolarlık hedefinize ulaştınız. Harika bir iş adamısınız. Ancak arkanızda bıraktığınız enkaza dönüp bir bakın.";
    } else {
        title.innerText = "İFLAS ETTİNİZ";
        title.style.color = "var(--neon-red)";
        desc.innerText = "Yeterince acımasız olamadınız. Kumarhane işi merhamet kaldırmaz. Kasa kapandı.";
    }

    stats.innerHTML = `
        <div style="margin-top: 2rem; font-size: 1.2rem; color: #fca5a5;">
            <p><strong>Mahvedilen Hayat Sayısı (İflas Edenler):</strong> ${state.ruinedLives}</p>
            <p><strong>Ulaşılan Gün:</strong> ${state.day}</p>
            <p><strong>Son İtibar Seviyesi:</strong> ${state.reputation}</p>
        </div>
    `;
}

// Action Buttons
document.getElementById('btn-marketing').onclick = () => {
    if (state.money >= 2000) {
        state.money -= 2000;
        state.marketingActive = true;
        state.reputation -= 5;
        logEvent("[TAKTİK] Eski bağımlılara 'Ücretsiz Fiş' mesajları atıldı. (Gerçekte: Kumara ara vermiş kişiler tetikleyici reklamlarla geri döndürülür.)", "bad");
        updateUI();
    } else {
        alert("Yeterli kasanız yok! ($2000 gerekli)");
    }
};

document.getElementById('btn-free-drinks').onclick = () => {
    if (state.money >= 500) {
        state.money -= 500;
        state.reputation -= 2;
        state.patrons.forEach(p => {
            p.addictionLevel += 10; // Drunk people make worse choices
        });
        logEvent("[TAKTİK] Tüm salona ücretsiz içki dağıtıldı. (Gerçekte: Alkol, beynin risk ve mantık hesaplama yeteneğini köreltir, cesaret verir.)", "bad");
        updateUI();
    } else {
        alert("Yeterli kasanız yok! ($500 gerekli)");
    }
};

// New Tactics and Upgrades
document.getElementById('btn-ldw').onclick = (e) => {
    state.ldwActive = !state.ldwActive;
    e.currentTarget.classList.toggle('active-tactic');
    if (state.ldwActive) {
        logEvent("[TAKTİK] Sahte Kazanç (LDW) algoritmaları açıldı. (Gerçekte: Makineler, oyuncu yatırdığından daha azını kazansa bile zafer müzikleri çalarak beyni kandırır).", "evil");
    } else {
        logEvent("LDW algoritmaları kapatıldı.", "system");
    }
};

document.getElementById('btn-scent').onclick = (e) => {
    state.scentActive = !state.scentActive;
    e.currentTarget.classList.toggle('active-tactic');
    if (state.scentActive) {
        logEvent("[TAKTİK] Havalandırmalara uyarıcı koku basılıyor. (Gerçekte: Belirli esanslar uyanıklığı, risk alma iştahını ve bahis miktarını artırır).", "evil");
    } else {
        logEvent("Özel havalandırma kapatıldı.", "system");
    }
};

document.getElementById('btn-vip-host').onclick = (e) => {
    state.vipHostActive = !state.vipHostActive;
    e.currentTarget.classList.toggle('active-tactic');
    if (state.vipHostActive) {
        logEvent("[TAKTİK] Balinalara (VIP) özel Hostlar atandı. (Gerçekte: Kişisel ilişki ve sahte arkadaşlık kurularak oyuncunun masadan kalkması engellenir).", "evil");
    } else {
        logEvent("VIP Hostlar geri çekildi.", "system");
    }
};

document.getElementById('upg-clocks').onclick = (e) => {
    if (!state.clocksRemoved && state.money >= 5000) {
        state.money -= 5000;
        state.clocksRemoved = true;
        e.currentTarget.classList.add('purchased');
        e.currentTarget.querySelector('strong').innerText = "Saatler Kaldırıldı";
        logEvent("[YÜKSELTME] Saatler ve pencereler kaldırıldı. (Gerçekte: Müşterilerin zaman algısı kaybolur ve 'Sensory Bubble' içinde hapsolurlar).", "bad");
        updateUI();
    } else if (!state.clocksRemoved) alert("Yeterli kasa yok! ($5.000)");
};

document.getElementById('upg-maze').onclick = (e) => {
    if (!state.mazeLayout && state.money >= 10000) {
        state.money -= 10000;
        state.mazeLayout = true;
        e.currentTarget.classList.add('purchased');
        e.currentTarget.querySelector('strong').innerText = "Labirent Zemin";
        logEvent("[YÜKSELTME] Zemin düzeni labirente çevrildi. (Gerçekte: Çıkışı bulmak zorlaştırılır ve yol üzerine cazip oyunlar konur).", "bad");
        updateUI();
    } else if (!state.mazeLayout) alert("Yeterli kasa yok! ($10.000)");
};

document.getElementById('upg-cashless').onclick = (e) => {
    if (!state.cashlessSystem && state.money >= 15000) {
        state.money -= 15000;
        state.cashlessSystem = true;
        e.currentTarget.classList.add('purchased');
        e.currentTarget.querySelector('strong').innerText = "Nakitsiz Sistem";
        logEvent("[YÜKSELTME] Kredi/Fiş sistemine geçildi. Müşteriler borca girebilir. (Gerçekte: Dijital para, fiziksel paranın verdiği harcama acısını yok eder).", "evil");
        updateUI();
    } else if (!state.cashlessSystem) alert("Yeterli kasa yok! ($15.000)");
};

document.getElementById('upg-cashback').onclick = (e) => {
    if (!state.cashbackActive && state.money >= 20000) {
        state.money -= 20000;
        state.cashbackActive = true;
        e.currentTarget.classList.add('purchased');
        e.currentTarget.querySelector('strong').innerText = "Kayıp İadesi Aktif";
        logEvent("[YÜKSELTME] Kayıp iadesi başladı. İflas edenlere %10 hediye edilecek. (Gerçekte: Beyni manipüle ederek para kaybetmeyi, ödül almakla eşleştirir).", "evil");
        updateUI();
    } else if (!state.cashbackActive) alert("Yeterli kasa yok! ($20.000)");
};

document.getElementById('upg-machine-zone').onclick = (e) => {
    if (!state.machineZoneActive && state.money >= 30000) {
        state.money -= 30000;
        state.machineZoneActive = true;
        e.currentTarget.classList.add('purchased');
        e.currentTarget.querySelector('strong').innerText = "Makine Zonu Hızlandırıcısı";
        logEvent("[YÜKSELTME] Müşteriler 'Makine Zonu' adı verilen transa sokuluyor. (Gerçekte: İnsanlar ortamdan kopar ve sadece tuşlara çok yüksek hızla basmaya odaklanırlar. Bahis hızı 3 katına çıkar).", "evil");
        updateUI();
    } else if (!state.machineZoneActive) alert("Yeterli kasa yok! ($30.000)");
};

document.getElementById('upg-ai-slots').onclick = (e) => {
    if (!state.aiSlotsActive && state.money >= 50000) {
        state.money -= 50000;
        state.aiSlotsActive = true;
        e.currentTarget.classList.add('purchased');
        e.currentTarget.querySelector('strong').innerText = "Yapay Zeka Slotları";
        logEvent("[YÜKSELTME] Boğ ve Bırak algoritmaları devreye girdi. (Gerçekte: Makine, oyuncunun bırakıp gideceğini hissettiği an ona 1 el kazandırır, bağımlılığını tazeler ve ardından tekrar tüm parasını yutar).", "evil");
        updateUI();
    } else if (!state.aiSlotsActive) alert("Yeterli kasa yok! ($50.000)");
};

document.getElementById('upg-online-app').onclick = (e) => {
    if (!state.onlineAppActive && state.money >= 40000) {
        state.money -= 40000;
        state.onlineAppActive = true;
        e.currentTarget.classList.add('purchased');
        e.currentTarget.querySelector('strong').innerText = "Online Uygulama Aktif";
        logEvent("[YÜKSELTME] Kumarhanenin cep telefonu uygulaması yayınlandı. (Gerçekte: Evine dönen ve iflas etmiş bağımlılar gece yataklarında bile kumar oynamaya devam eder. Pasif gelir sağlar).", "evil");
        updateUI();
    } else if (!state.onlineAppActive) alert("Yeterli kasa yok! ($40.000)");
};

document.getElementById('upg-debt-collector').onclick = (e) => {
    if (!state.debtCollectorActive && state.money >= 75000) {
        state.money -= 75000;
        state.debtCollectorActive = true;
        e.currentTarget.classList.add('purchased');
        e.currentTarget.querySelector('strong').innerText = "Acımasız Tahsildar (Tefeci)";
        logEvent("[YÜKSELTME] Tefeci anlaşması yapıldı. (Gerçekte: Kredili sistemde borca giren müşterilerin tüm mallarına ve hayatlarına fahiş faizlerle el konulur).", "evil");
        updateUI();
    } else if (!state.debtCollectorActive) alert("Yeterli kasa yok! ($75.000)");
};

document.getElementById('upg-cashier-sludge').onclick = (e) => {
    if (!state.sludgeActive && state.money >= 12000) {
        state.money -= 12000;
        state.sludgeActive = true;
        e.currentTarget.classList.add('purchased');
        e.currentTarget.querySelector('strong').innerText = "Vezne Frikşonu Aktif";
        logEvent("[YÜKSELTME] Vezne kasten en uzağa taşındı ve personel azaltıldı. Müşteriler kasayı bulamayıp sıkılacak ve yolda gördükleri makinelere geri dönecekler.", "evil");
        updateUI();
    } else if (!state.sludgeActive) alert("Yeterli kasa yok! ($12.000)");
};

document.getElementById('upg-illusion').onclick = (e) => {
    if (!state.illusionActive && state.money >= 25000) {
        state.money -= 25000;
        state.illusionActive = true;
        e.currentTarget.classList.add('purchased');
        e.currentTarget.querySelector('strong').innerText = "Kontrol İllüzyonu Aktif";
        logEvent("[YÜKSELTME] Makinelere sahte karar butonları eklendi. Müşteriler kaderlerinin kendi ellerinde olduğuna inanıp daha hızlı oynamaya başladı.", "evil");
        updateUI();
    } else if (!state.illusionActive) alert("Yeterli kasa yok! ($25.000)");
};

document.getElementById('upg-reverse-withdraw').onclick = (e) => {
    if (!state.reverseWithdrawalActive && state.money >= 60000) {
        state.money -= 60000;
        state.reverseWithdrawalActive = true;
        e.currentTarget.classList.add('purchased');
        e.currentTarget.querySelector('strong').innerText = "Ters Para Çekme Aktif";
        logEvent("[YÜKSELTME] Ters Para Çekme yürürlükte! Kazanan müşterilerin ödemeleri 24 saat ertelenir, onlara 'İptal Et' tuşu verilir ve parayı tekrar kasaya gömmeleri beklenir.", "evil");
        updateUI();
    } else if (!state.reverseWithdrawalActive) alert("Yeterli kasa yok! ($60.000)");
};

// Tab Menu Logic
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        e.target.classList.add('active');
        const targetId = e.target.getAttribute('data-target');
        document.getElementById(targetId).classList.add('active');
    });
});

// Start Game Loop
setInterval(gameTick, 2500); // Slower tick: 2.5 seconds per game hour
logEvent("Sistem başlatıldı. Başlangıç kasası: $50,000", "system");
updateUI();
